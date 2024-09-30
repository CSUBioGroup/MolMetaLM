import numpy as np
import pandas as pd
import torch,time,os,pickle,random,re,gc
from torch import nn as nn
from nnLayer import *
from metrics import *
from sklearn.model_selection import StratifiedKFold,KFold
from torch.backends import cudnn
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
import torch.distributed
from functools import partial
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from joblib.externals.loky.backend.context import get_context
from utils import *
from tools import *

class BaseClassifier:
    def __init__(self, model):
        pass
    def calculate_y_logit(self, X):
        pass
    def calculate_y_prob(self, X):
        pass
    def calculate_y(self, X):
        pass
    def calculate_y_prob_by_iterator(self, dataStream):
        pass
    def calculate_y_by_iterator(self, dataStream):
        pass
    def calculate_loss(self, X, Y):
        pass
    def train(self, trainDataSet, validDataSet=None, otherDataSet=None, otherSampleNum=10000, doEvalTrain=False,
              batchSize=256, maxSteps=1000000, saveSteps=-1, evalSteps=100, earlyStop=10, beginSteps=-1,
              EMA=False, EMAdecay=0.9999, EMAse=16, scheduleLR=True,
              isHigherBetter=False, metrics="LOSS", report=["LOSS"], 
              savePath='model', shuffle=True, dataLoadNumWorkers=0, pinMemory=False, 
              ignoreIdx=-100, warmupSteps=0, SEED=0, tensorboard=False, prefetchFactor=2, bleuCommend=None):
        if self.DDP:
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(f'./logs/{tensorboard}')
            self.writer = writer

        metrictor = self.metrictor if hasattr(self, "metrictor") else Metrictor()
        device = next(self.model.parameters()).device
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        stop = False

        if scheduleLR:
            decaySteps = maxSteps - warmupSteps
            # schedulerRLR = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda i:i/warmupSteps if i<warmupSteps else (decaySteps-(i-warmupSteps))/decaySteps)
            schedulerRLR = get_cosine_schedule_with_warmup(self.optimizer, num_training_steps=maxSteps, num_warmup_steps=warmupSteps)

        if self.DDP:
            trainSampler = torch.utils.data.distributed.DistributedSampler(trainDataSet, shuffle=True, seed=SEED)
        else:
            trainSampler = torch.utils.data.RandomSampler(trainDataSet)
        # trainSampler = torch.utils.data.RandomSampler(trainDataSet)
        trainStream = DataLoader(trainDataSet, batch_size=batchSize, collate_fn=self.collateFunc, sampler=trainSampler, drop_last=True,
                                 pin_memory=pinMemory, num_workers=dataLoadNumWorkers, prefetch_factor=prefetchFactor)

        if validDataSet is not None:
            if self.DDP:
                validSampler = torch.utils.data.distributed.DistributedSampler(validDataSet, shuffle=False)
            else:
                validSampler = torch.utils.data.SequentialSampler(validDataSet)
            # validSampler = torch.utils.data.SequentialSampler(validDataSet)
            validStream = DataLoader(validDataSet, batch_size=batchSize, collate_fn=self.collateFunc, sampler=validSampler, drop_last=False, 
                                     pin_memory=pinMemory, num_workers=dataLoadNumWorkers, prefetch_factor=prefetchFactor)       
     
        mtc,bestMtc,stopSteps = 0.0,0.0 if isHigherBetter else 9999999999,0
        st = time.time()
        e,locStep = 0,-1
        ema = None

        # restore the state
        if beginSteps>0:
            for i in range(beginSteps):
                schedulerRLR.step()
            locStep = beginSteps

        while True:
            e += 1
            self.locEpoch = e
            if EMA and e>=EMAse and ema is None:
                print('Start EMA...')
                ema = EMAer(self.model, EMAdecay) # 0.9999
                ema.register()
            print(f"Preparing the epoch {e}'s data...")
            if hasattr(trainDataSet, 'init'): 
                trainDataSet.init(e)
            # if otherDataSet is not None:
            #    sampleIdx = random.sample(range(len(otherDataSet)), otherSampleNum)
            #    trainDS = torch.utils.data.ConcatDataset([trainDataSet, torch.utils.data.Subset(otherDataSet, sampleIdx)])
            #    if self.DDP:
            #        trainSampler = torch.utils.data.distributed.DistributedSampler(trainDS, shuffle=True, seed=SEED)
            #    else:
            #        trainSampler = torch.utils.data.RandomSampler(trainDS)
            #    trainStream = DataLoader(trainDS, batch_size=batchSize, collate_fn=self.collateFunc, sampler=trainSampler, drop_last=True,
            #                             pin_memory=pinMemory, num_workers=dataLoadNumWorkers, prefetch_factor=prefetchFactor)
            if self.DDP:
                trainStream.sampler.set_epoch(e)
            pbar = tqdm(trainStream)
            self.to_train_mode()
            for data in pbar:
                data = dict_to_device(data, device=device)
                loss = self._train_step(data)
                # del data
                # gc.collect()
                if EMA and ema is not None:
                    ema.update()
                if scheduleLR:
                    schedulerRLR.step()
                if isinstance(loss, dict):
                    pbar.set_description(f"Training Loss: { {k:round(float(loss[k]),3) for k in loss} }; Learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']:.6f}; Progress: {locStep/maxSteps:.3f}; Stop round: {stopSteps}")
                else:
                    pbar.set_description(f"Training Loss: {loss:.3f}; Learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']:.6f}; Progress: {locStep/maxSteps:.3f}; Stop round: {stopSteps}")
                locStep += 1
                self.locStep = locStep
                if locStep>maxSteps:
                    print(f'Reach the max steps {maxSteps}... break...')
                    stop = True
                    break
                if (validDataSet is not None) and (locStep%evalSteps==0):
                    if EMA and ema is not None:
                        ema.apply_shadow()
                    print(f'========== Step:{locStep:5d} ==========')

                    self.to_eval_mode()
                    if doEvalTrain:
                        print(f'[Total Train]', end='')
                        if metrics == 'CUSTOMED':
                            mtc = self.calculate_CUSTOMED_metrics(trainStream)
                        else:
                            res = self.calculate_metrics_by_iterator(trainStream, metrictor, ignoreIdx, report)
                            metrictor.show_res(res)
                            mtc = res[metrics]

                    #data = self.calculate_y_prob_by_iterator(DataLoader(trainDataSet, batch_size=batchSize, shuffle=False, num_workers=dataLoadNumWorkers, pin_memory=pinMemory, collate_fn=self.collateFunc, sampler=trainSampler))
                    #print(Y_pre.shape, Y.shape, type(Y_pre), type(Y))
                    #metrictor.set_data(data, ignoreIdx)
                    #print(f'[Total Train]',end='')
                    #metrictor(report)

                    print(f'[Total Valid]',end='')
                    if metrics == 'CUSTOMED':
                        mtc = self.calculate_CUSTOMED_metrics(validStream)
                    else:
                        res = self.calculate_metrics_by_iterator(validStream, metrictor, ignoreIdx, report)
                        metrictor.show_res(res)
                        mtc = res[metrics]
                    if self.DDP:
                        mtc = torch.tensor([mtc], dtype=torch.float32, device='cuda')
                        mtc_list = [torch.zeros_like(mtc) for i in range(world_size)]
                        torch.distributed.all_gather(mtc_list, mtc)
                        mtc = torch.cat(mtc_list).mean().detach().cpu().item()
                    if ((self.DDP and torch.distributed.get_rank() == 0) or not self.DDP):
                        # if tensorboard:
                        #     writer.add_scalar(metrics, mtc, locStep)
                        print('=================================')

                        if saveSteps>0 and locStep%saveSteps==0:
                            if (self.DDP and torch.distributed.get_rank() == 0) or not self.DDP:
                                self.save("%s_acc%.3f_s%d.pkl"%(savePath,mtc,locStep), e+1, mtc)

                        if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                            if (self.DDP and torch.distributed.get_rank() == 0) or not self.DDP:                
                                print(f'Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!')
                                bestMtc = mtc
                                self.save("%s.pkl"%(savePath), e+1, bestMtc)
                            stopSteps = 0
                        else:
                            stopSteps += 1
                            if earlyStop>0 and stopSteps>=earlyStop:
                                print(f'The val {metrics} has not improved for more than {earlyStop} steps in epoch {e+1}, stop training.')
                                stop = True
                                break

                        if EMA and ema is not None:
                            ema.restore()
                self.to_train_mode()
            if stop:
                break
        if tensorboard:
            writer.close()
        if (self.DDP and torch.distributed.get_rank() == 0) or not self.DDP:
            if EMA and ema is not None:
                ema.apply_shadow()
            self.load("%s.pkl"%savePath)
            self.to_eval_mode()
            os.rename("%s.pkl"%savePath, "%s_%s.pkl"%(savePath, "%.3lf"%bestMtc))
            print(f'============ Result ============')
            if doEvalTrain:
                print(f'[Total Train]',end='')
                if metrics == 'CUSTOMED':
                    res = self.calculate_CUSTOMED_metrics(trainStream)
                else:
                    res = self.calculate_metrics_by_iterator(trainStream, metrictor, ignoreIdx, report)
                    metrictor.show_res(res)
            print(f'[Total Valid]',end='')
            if metrics == 'CUSTOMED':
                res = self.calculate_CUSTOMED_metrics(validStream)
            else:
                res = self.calculate_metrics_by_iterator(validStream, metrictor, ignoreIdx, report)
                metrictor.show_res(res)
            print(f'================================')
            return res
    def to_train_mode(self):
        self.model.train()  #set the module in training mode
        if self.collateFunc is not None:
            self.collateFunc.train = True
    def to_eval_mode(self):
        self.model.eval()
        if self.collateFunc is not None:
            self.collateFunc.train = False
    def _train_step(self, data):
        loss = self.calculate_loss(data)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach().cpu().data.numpy()
    def save(self, path, epochs, bestMtc=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc, 'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None):
        parameters = torch.load(path, map_location=map_location)
        self.model.load_state_dict(parameters['model'])
        if 'optimizer' in parameters:
            self.optimizer.load_state_dict(parameters['optimizer'])
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc'] if parameters['bestMtc'] is not None else 0.00))

from torch.cuda.amp import autocast, GradScaler
class Seq2SeqLanguageModel(BaseClassifier):
    def __init__(self, model, criterion=None, optimizer=None, collateFunc=None, ignoreIdx=-100, 
                 AMP=False, DDP=False, FGM=False):
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignoreIdx) if criterion is None else criterion
        self.optimizer = torch.optim.AdamW(model.parameters()) if optimizer is None else optimizer
        self.collateFunc = collateFunc
        self.AMP,self.DDP,self.FGM = AMP,DDP,FGM
        if AMP:
            self.scaler = GradScaler()
    def calculate_y_logit(self, data, predict=False):
        if self.AMP:
            with autocast():
                return self.model(data, predict)
        else:
            return self.model(data, predict)
    def calculate_y_prob(self, data):
        Y_pre = self.calculate_y_logit(data, predict=True)['y_logit']
        return {'y_prob':F.softmax(Y_pre, dim=-1)}
    def calculate_y_pre(self, data):
        Y_pre = self.calculate_y_logit(data, predict=True)['y_logit']
        return {'y_pre':torch.argmax(Y_pre, dim=-1)}
    def generate(self, data, beamwidth=4):
        if self.AMP:
            with autocast():
                res = self.model.beamsearch(data, beamwidth=beamwidth)
        else:
            res = self.model.beamsearch(data, beamwidth=beamwidth)
        res['y_pre'] = res['y_pre']
        sortIdx = res['scoreArr'].argsort(dim=1, descending=True)
        return {'y_pre':res['y_pre'][torch.arange(len(sortIdx)).unsqueeze(dim=-1),sortIdx], 'scoreArr':res['scoreArr'][torch.arange(len(sortIdx)).unsqueeze(dim=-1),sortIdx]}
    def calculate_loss_by_iterator(self, dataStream):
        loss,cnt = 0,0
        for data in dataStream:
            loss += self.calculate_loss(data) * len(data['tokenizedSeqArr'])
            cnt += len(data['tokenizedSeqArr'])
        return loss / cnt
    def calculate_y_prob_by_iterator(self, dataStream):
        device = next(self.model.parameters()).device
        YArr,Y_probArr,maskIdxArr = [],[],[]
        for data in tqdm(dataStream):
            data = dict_to_device(data, device=device)
            Y_prob,Y,maskIdx = self.calculate_y_prob(data)['y_prob'][:,:-1].detach().cpu().data.numpy(),data['targetTknArr'][:,1:].detach().cpu().data.numpy(),data['tmaskPAD'][:,1:].detach().cpu().data.numpy()
            YArr.append(Y)
            Y_probArr.append(Y_prob)
            maskIdxArr.append(maskIdx)
        YArr,Y_probArr,maskIdxArr = np.vstack(YArr).astype('int32'),np.vstack(Y_probArr).astype('float32'),np.vstack(maskIdxArr).astype('bool')
        return {'y_prob':Y_probArr, 'y_true':YArr, 'mask_idx':maskIdxArr}
    
    def calculate_y_pre_by_iterator(self, dataStream):
        device = next(self.model.parameters()).device
        YArr,Y_preArr,maskIdxArr = [],[],[]
        for data in tqdm(dataStream):
            data = dict_to_device(data, device=device)
            Y_pre,Y,maskIdx = self.calculate_y_pre(data)['y_pre'][:,:-1].detach().cpu().data.numpy(),data['targetTknArr'][:,1:].detach().cpu().data.numpy(),data['tMaskPAD'][:,1:].detach().cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
            maskIdxArr.append(maskIdx)
        YArr,Y_preArr,maskIdxArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('int32'),np.vstack(maskIdxArr).astype('bool')
        return {'y_pre':Y_preArr, 'y_true':YArr, 'mask_idx':maskIdxArr}
    def calculate_metrics_by_iterator(self, dataStream, metrictor, ignoreIdx, report):
        # if self.collateFunc is not None:
        #     self.collateFunc.train = True
        device = next(self.model.parameters()).device
        YArr,Y_preArr,maskIdxArr = [],[],[]
        res,cnt = {},0
        for data in tqdm(dataStream):
            data = dict_to_device(data, device=device)
            with torch.no_grad():
                Y_pre,Y,maskIdx = self.calculate_y(data)['y_pre'][:,:-1].detach().cpu().data.numpy().astype('int32'),data['targetTknArr'][:,1:].detach().cpu().data.numpy().astype('int32'),data['tMaskPAD'][:,1:].detach().cpu().data.numpy().astype('bool')
            batchData = {'y_pre':Y_pre, 'y_true':Y, 'mask_idx':maskIdx}
            metrictor.set_data(batchData, ignore_index=ignoreIdx)
            batchRes = metrictor(report, isPrint=False)
            for k in batchRes:
                res.setdefault(k, 0)
                res[k] += batchRes[k]*len(Y_pre)
            cnt += len(Y_pre)
        return {k:res[k]/cnt for k in res}
    def calculate_loss(self, data):
        out = self.calculate_y_logit(data)
        Y = data['targetTknArr'][:,1:].reshape(-1)
        Y_logit = out['y_logit'][:,:-1].reshape(len(Y),-1)
        maskIdx = data['tMaskPAD'][:,1:].reshape(-1)

        return self.criterion(Y_logit[maskIdx], Y[maskIdx]) # [maskIdx]
    def _train_step(self, data):
        self.optimizer.zero_grad()
        if self.AMP:
            with autocast():
                loss = self.calculate_loss(data)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.calculate_loss(data)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
        return loss.detach().cpu().data.numpy()
    def save(self, path, epochs, bestMtc=None):
        if self.DDP:
            model = self.model.module.state_dict()
        else:
            model = self.model.state_dict()
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc, 'model':model, 'optimizer':self.optimizer.state_dict()}
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None):
        parameters = torch.load(path, map_location=map_location)
        if self.DDP:
            self.model.module.load_state_dict(parameters['model'])
        else:
            self.model.load_state_dict(parameters['model'])
        if 'optimizer' in parameters:
            try:
                self.optimizer.load_state_dict(parameters['optimizer'])
            except:
                print("Warning! Cannot restore the optimizer parameters...")
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc'] if parameters['bestMtc'] is not None else 0.00))

from transformers import AutoTokenizer, AutoModel, AutoModel, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForMaskedLM
from transformers import BertModel

class HuggingfaceSeq2SeqLanguageModel(Seq2SeqLanguageModel):
    def __init__(self, model, criterion=None, optimizer=None, collateFunc=None, ignoreIdx=-100, 
                 AMP=False, DDP=False, FGM=False, FGMeps=1., metrictor=None):
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignoreIdx) if criterion is None else criterion
        self.optimizer = torch.optim.AdamW(model.parameters()) if optimizer is None else optimizer
        self.collateFunc = collateFunc
        self.AMP,self.DDP,self.FGM = AMP,DDP,FGM
        if metrictor is not None:
            self.metrictor = metrictor
        if AMP:
            self.scaler = GradScaler()
        if FGM:
            self.fgm = FGMer(model, emb_name=['shared']) # word_embedding
            self.fgmEps = FGMeps
    def _train_step(self, data):
        self.optimizer.zero_grad()
        if self.AMP:
            with autocast():
                loss = self.calculate_loss(data)
                if isinstance(loss, dict):
                    loss_all = 0
                    for k in loss:
                        loss_all += loss[k]
                else:
                    loss_all = loss
            self.scaler.scale(loss_all).backward()
            if self.FGM and self.locEpoch >= self.FGMse: # 对抗损失
                self.fgm.attack(self.fgmEps)
                with autocast():
                    lossAdv = self.calculate_loss(data)
                self.scaler.scale(lossAdv).backward()
                self.fgm.restore()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.calculate_loss(data)
            if isinstance(loss, dict):
                loss_all = 0
                for k in loss:
                    loss_all += loss[k]
            else:
                loss_all = loss
            loss_all.backward()
            if self.FGM:
                self.fgm.attack(self.fgmEps)
                lossAdv = self.calculate_loss(data)
                lossAdv.backward()
                self.fgm.fgm.restore()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
        return loss # .detach().cpu().data.numpy()
    def calculate_y_logit(self, data, predict=False):
        if self.AMP:
            with autocast():
                return self.model(data, predict)
        else:
            return self.model(data, predict)
    def calculate_tkn_pre_by_iterator(self, dataStream):
        device = next(self.model.parameters()).device
        YArr,Y_preArr = [],[]
        for data in tqdm(dataStream):
            data = dict_to_device(data, device=device)
            with torch.no_grad():
                Y_pre,Y = self.calculate_y_pre(data)['y_pre'].detach().cpu().data.numpy().astype('int32').tolist(),data['batch']['labels'].detach().cpu().data.numpy().astype('int32').tolist()
            YArr += self.collateFunc.tokenizer.batch_decode(Y, skip_special_tokens=True)
            Y_preArr += [i[i.find('[SOS]')+6:i.find('[EOS]')] for i in self.collateFunc.tokenizer.batch_decode(Y_pre, skip_special_tokens=False)]
        return {'target_pre':Y_preArr, 'target_true':YArr}
    def calculate_loss(self, data):
        out = self.calculate_y_logit(data)
        # return out['loss']
        Y = data['batch']['labels'][:,1:].reshape(-1)
        Y_logit = out['y_logit'][:,:-1].reshape(len(Y),-1)
        return self.criterion(Y_logit, Y)
    def calculate_loss_by_iterator(self, dataStream):
        loss,cnt = 0,0
        for data in dataStream:
            loss += self.calculate_loss(data) * len(data['tokenizedSeqArr'])
            cnt += len(data['tokenizedSeqArr'])
        return loss / cnt
    def calculate_metrics_by_iterator(self, dataStream, metrictor, ignoreIdx, report):
        # if self.collateFunc is not None:
        #     self.collateFunc.train = True
        device = next(self.model.parameters()).device
        YArr,Y_preArr,maskIdxArr = [],[],[]
        res,cnt = {},0
        for data in tqdm(dataStream):
            data = dict_to_device(data, device=device)
            with torch.no_grad():
                # Y_pre,Y = self.calculate_y_pre(data)['y_pre'].detach().cpu().data.numpy().astype('int32'),data['labels'].detach().cpu().data.numpy().astype('int32')
                Y_prob,Y = self.calculate_y_prob(data)['y_prob'].detach().cpu().data.numpy().astype('float32'),data['batch']['labels'].detach().cpu().data.numpy().astype('int32')
                Y_prob,Y = Y_prob[:,:-1],Y[:,1:] # shift predictions and labels
                maskIdx = np.ones(Y.shape, dtype=bool)
                batchData = {'y_prob':Y_prob, 'y_true':Y, 'mask_idx':maskIdx}
                metrictor.set_data(batchData, ignore_index=ignoreIdx)
                batchRes = metrictor(report, isPrint=False)
            for k in batchRes:
                res.setdefault(k, 0)
                res[k] += batchRes[k]*len(Y)
            cnt += len(Y)
        return {k:res[k]/cnt for k in res}
    def calculate_item_loss_by_iterator(self, dataStream, ignoreIdx):
        device = next(self.model.parameters()).device
        criterion = nn.CrossEntropyLoss(ignore_index=ignoreIdx, reduction='none')
        strategies,losses = [],[]
        for data in tqdm(dataStream):
            data = dict_to_device(data, device=device)
            with torch.no_grad():
                Y_logit,Y = self.calculate_y_logit(data)['y_logit'].detach().cpu(),data['batch']['labels'].detach().cpu()
                Y_logit,Y = Y_logit[:,:-1],Y[:,1:] # shift predictions and labels
                B,L = Y.shape
                loss = criterion(Y_logit.reshape(B*L,-1), Y.reshape(-1)).reshape(B,L)
                loss = loss.sum(dim=1) / (Y!=ignoreIdx).sum(dim=1)
                losses.append(loss.data.numpy().astype('float32'))
                strategies += data['strategies']
        return {'strategies':strategies, 'losses':np.hstack(losses)}
    def calculate_y_pre(self, data):
        tmp = self.calculate_y_logit(data, predict=True)
        if 'y_pre' in tmp:
            y_pre = tmp['y_pre']
        else:
            y_pre = torch.argmax(tmp['y_logit'], dim=-1)
        return {'y_pre':y_pre}
    def calculate_y_prob(self, data):
        Y_pre = self.calculate_y_logit(data, predict=True)['y_logit']
        return {'y_prob':F.softmax(Y_pre, dim=-1)}
    def generate(self, data, beamwidth=None, lengthPenalty=None):
        if self.AMP:
            with autocast():
                res = self.model(data, predict=True, num_beams=beamwidth, length_penalty=lengthPenalty)
        else:
            res = self.model(data, predict=True, num_beams=beamwidth, length_penalty=lengthPenalty)
        return res
    def calculate_CUSTOMED_metrics(self, dataStream):
        bak = self.model.alwaysTrain
        self.model.alwaysTrain = False
        def convert2float(s):
            s = s.replace('[SEP]','').replace(' ','')
            try:
                if re.match("^[+-]*\d+\.\d*;$",s):
                    return float(s[:-1])
                else:
                    return np.nan
            except:
                return np.nan
        tkn_pre = self.calculate_tkn_pre_by_iterator(dataStream)
        vPre = np.array([convert2float(i) for i in tkn_pre['target_pre']], dtype=np.float32)
        vTrue = np.array([convert2float(i) for i in tkn_pre['target_true']], dtype=np.float32)
        isValid = ~np.isnan(vPre)
        vPre,vTrue = vPre[isValid],vTrue[isValid]
        MSE = np.mean((vPre-vTrue)**2)
        R2 = ((np.mean(vPre*vTrue) - np.mean(vPre)*np.mean(vTrue)) / np.sqrt((np.mean(vPre**2)-np.mean(vPre)**2) * (np.mean(vTrue**2)-np.mean(vTrue)**2)))**2
        VR  = sum(isValid)/len(isValid)
        SCORE = MSE/VR
        print(f'Valid Rate: {VR:.3f}; MSE: {MSE:.3f}; R2: {R2:.3f}; SCORE: {SCORE:.3f}')
        self.model.alwaysTrain = bak
        return SCORE

class HuggingfaceSeq2SeqLanguageModel_ForRegression(HuggingfaceSeq2SeqLanguageModel):
    def __init__(self, model, criterion=None, optimizer=None, collateFunc=None, 
                 AMP=False, DDP=False, 
                 FGM=False, FGMeps=1., FGMse=-1,
                 metrictor=None):
        self.model = model
        self.criterion = nn.MSELoss() if criterion is None else criterion
        self.optimizer = torch.optim.AdamW(model.parameters()) if optimizer is None else optimizer
        self.collateFunc = collateFunc
        self.AMP,self.DDP,self.FGM = AMP,DDP,FGM
        if metrictor is not None:
            self.metrictor = metrictor
        if AMP:
            self.scaler = GradScaler()
        if FGM:
            self.fgm = FGMer(model, emb_name=['shared']) # word_embedding
            self.fgmEps = FGMeps
            self.FGMse = FGMse
    def calculate_loss(self, data):
        out = self.model(data)

        # return out['loss']
        Y = data['y'].reshape(-1)
        if 'y_logit_list' not in out:
            Y_logit = out['y_logit'].reshape(-1)
            return self.criterion(Y_logit, Y)
        else:
            loss = 0
            for y_logit in out['y_logit_list']:
                loss += self.criterion(y_logit.reshape(-1), Y)
            return loss
    def calculate_y_pre(self, data):
        tmp = self.model(data)
        return {'y_pre':tmp['y_logit']}
    def calculate_y_pre_by_iterator(self, dataStream):
        device = next(self.model.parameters()).device
        YArr,Y_preArr = [],[]
        for data in tqdm(dataStream):
            data = dict_to_device(data, device=device)
            Y_pre,Y = self.calculate_y_pre(data)['y_pre'].detach().cpu().data.numpy().reshape(-1),data['y'].detach().cpu().data.numpy().reshape(-1)
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.hstack(YArr).astype('float32'),np.hstack(Y_preArr).astype('float32')
        return {'y_pre':Y_preArr, 'y_true':YArr}
    def calculate_metrics_by_iterator(self, dataStream, metrictor, ignoreIdx, report, 
                                      TTA_num=1, TTA_obj=None, isHigherBetter=False, tol=1e-3):
        device = next(self.model.parameters()).device
        if TTA_num==1:
            res = self.calculate_y_pre_by_iterator(dataStream)
        else: # using TTA for prediction
            yp,bestYp,bestMtc = 0,None,-1 if isHigherBetter else np.inf
            train = dataStream.collate_fn.train
            for i in range(TTA_num):
                res = self.calculate_y_pre_by_iterator(dataStream)
                yp += res['y_pre']
                res['y_pre'] = yp / (i+1)
                if TTA_obj is not None:
                    metrictor.set_data(res, ignore_index=ignoreIdx)
                    locMtc = metrictor(report, isPrint=False)[TTA_obj]

                    if (isHigherBetter and locMtc>bestMtc) or (not isHigherBetter and locMtc<bestMtc):
                        bestYp,bestMtc = res['y_pre'],locMtc
                        print(f'Better Metrics of {bestMtc} at TTA iter {i+1}...')
                    elif np.abs(locMtc-bestMtc)>tol:
                        print(f'Worse Metrics of {locMtc} at TTA iter {i+1}...exit TTA...')
                        break
                dataStream.collate_fn.train = True # turn on random SMILES mode
            dataStream.collate_fn.train = train # restore the train state
            res['y_pre'] = bestYp
        if self.collateFunc.normVec is not None:
            mean,std = self.collateFunc.normVec
            yp,yt = res['y_pre'].reshape(-1,len(mean)),res['y_true'].reshape(-1,len(mean))
            yp,yt = yp*std+mean,yt*std+mean
            res['y_pre'],res['y_true'] = yp,yt
        metrictor.set_data(res, ignore_index=ignoreIdx)
        return metrictor(report, isPrint=False)

class HuggingfaceSeq2SeqLanguageModel_ForClassification(HuggingfaceSeq2SeqLanguageModel):
    def __init__(self, model, criterion=None, optimizer=None, collateFunc=None, 
                 AMP=False, DDP=False, 
                 FGM=False, FGMeps=1., FGMse=-1,
                 metrictor=None,
                 multilabel=False):
        self.model = model
        self.criterion = (nn.CrossEntropyLoss() if not multilabel else nn.MultiLabelSoftMarginLoss()) if criterion is None else criterion
        self.optimizer = torch.optim.AdamW(model.parameters()) if optimizer is None else optimizer
        self.collateFunc = collateFunc
        self.AMP,self.DDP,self.FGM = AMP,DDP,FGM
        self.multilabel = multilabel
        if metrictor is not None:
            self.metrictor = metrictor
        if AMP:
            self.scaler = GradScaler()
        if FGM:
            self.fgm = FGMer(model, emb_name=['shared']) # word_embedding
            self.fgmEps = FGMeps
            self.FGMse = FGMse
    def calculate_loss(self, data):
        out = self.model(data)

        # return out['loss']
        if (len(data['y'].shape)==2 and data['y'].shape[1]==1 and not self.multilabel): Y = data['y'].reshape(-1).long()
        else: Y = data['y'].reshape(out['y_logit'].shape).long()
        if 'y_logit_list' not in out:
            Y_logit = out['y_logit'].reshape(len(Y),-1)
            return self.criterion(Y_logit, Y)
        else:
            loss = 0
            for y_logit in out['y_logit_list']:
                loss += self.criterion(y_logit.reshape(len(Y),-1), Y)
            # print(out['y_logit'].reshape(-1))
            # print(Y)
            # print()
            return loss
    def calculate_y_prob(self, data):
        tmp = self.model(data)
        if self.multilabel:
            return {'y_prob':F.sigmoid(tmp['y_logit'])}
        else:
            return {'y_prob':F.softmax(tmp['y_logit'], dim=-1)}
    def calculate_y_prob_by_iterator(self, dataStream):
        device = next(self.model.parameters()).device
        YArr,Y_probArr = [],[]
        for data in tqdm(dataStream):
            data = dict_to_device(data, device=device)
            Y_prob,Y = self.calculate_y_prob(data)['y_prob'].detach().cpu().data.numpy(),data['y'].detach().cpu().data.numpy()
            YArr.append(Y)
            Y_probArr.append(Y_prob)

        YArr,Y_probArr = np.concatenate(YArr, axis=0).astype('int32'),np.concatenate(Y_probArr, axis=0).astype('float32')
        return {'y_prob':Y_probArr, 'y_true':YArr}

    def calculate_metrics_by_iterator(self, dataStream, metrictor, ignoreIdx, report, 
                                      TTA_num=1, TTA_obj=None, isHigherBetter=True, tol=1e-3):
        device = next(self.model.parameters()).device
        if TTA_num==1:
            res = self.calculate_y_prob_by_iterator(dataStream)
        else: # using TTA for prediction
            yp,bestYp,bestMtc = 0,None,-1 if isHigherBetter else np.inf
            train = dataStream.collate_fn.train
            for i in range(TTA_num):
                res = self.calculate_y_prob_by_iterator(dataStream)
                yp += res['y_prob']
                res['y_prob'] = yp / (i+1)
                if TTA_obj is not None:
                    metrictor.set_data(res, ignore_index=ignoreIdx, multilabel=self.multilabel)
                    locMtc = metrictor(report, isPrint=False)[TTA_obj]

                    if (isHigherBetter and locMtc>bestMtc) or (not isHigherBetter and locMtc<bestMtc):
                        bestYp,bestMtc = res['y_prob'],locMtc
                        print(f'Better Metrics of {bestMtc} at TTA iter {i+1}...')
                    elif np.abs(locMtc-bestMtc)>tol:
                        print(f'Worse Metrics of {locMtc} at TTA iter {i+1}...exit TTA...')
                        res['y_prob'] = bestYp
                        break
                dataStream.collate_fn.train = True # turn on random SMILES mode
            dataStream.collate_fn.train = train
            res['y_prob'] = bestYp
        metrictor.set_data(res, ignore_index=ignoreIdx, multilabel=self.multilabel)
        return metrictor(report, isPrint=False)

class MolMetaLM(nn.Module):
    def __init__(self, config, tkn2id, maxGenLen=128, 
                 num_beams=1, length_penalty=1.0):
        super(MolMetaLM, self).__init__()
        self.backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        self.backbone.config.forced_eos_token_id = self.backbone.config.eos_token_id = tkn2id['[EOS]']
        self.backbone.config.bos_token_id = tkn2id['[SOS]']
        self.backbone.config.pad_token_id = tkn2id['[PAD]']
        self.backbone.config.mask_token_id = tkn2id['[MASK]']
        self.backbone.config.forced_bos_token_id = None
        self.backbone.config.decoder_start_token_id = tkn2id['[SOS]']

        self.backbone.resize_token_embeddings(0)
        self.backbone.resize_token_embeddings(len(tkn2id))
        try:
            self.backbone.transformer.wte.padding_idx = tkn2id['[PAD]']
        except:
            self.backbone.model.embed_tokens.padding_idx = tkn2id['[PAD]']

        self.maxGenLen = maxGenLen
        self.num_beams,self.length_penalty = num_beams,length_penalty
        self.alwaysTrain = False

    def forward(self, data, predict=False, num_beams=None, length_penalty=None):
        if num_beams is None:
            num_beams = self.num_beams
        if length_penalty is None:
            length_penalty = self.length_penalty

        batch = data['batch']
        if predict and not self.alwaysTrain:
            return {'y_pre': self.backbone.generate(**({k:batch[k] for k in batch if k!='labels'}),
                                                    max_length=self.maxGenLen, num_beams=num_beams, length_penalty=length_penalty, 
                                                    no_repeat_ngram_size=0)}
        else:
            tmp = dict(self.backbone(**batch))
            tmp['y_logit'] = tmp.pop('logits')
            return tmp

class MolMetaLM_finetune_Alpha(nn.Module): # for GPT
    def __init__(self, config, tkn2id, classNum=1, fcSize=1024, classifierType='Linear', 
                 norm=True, dropout=0.1, reduction='pool_ensemble', use_fea_list=['max','mean']):
        super(MolMetaLM_finetune_Alpha, self).__init__()
        self.backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        self.backbone.config.forced_eos_token_id = self.backbone.config.eos_token_id = tkn2id['[EOS]']
        self.backbone.config.bos_token_id = tkn2id['[SOS]']
        self.backbone.config.pad_token_id = tkn2id['[PAD]']
        self.backbone.config.mask_token_id = tkn2id['[MASK]']
        self.backbone.config.forced_bos_token_id = None
        self.backbone.config.decoder_start_token_id = tkn2id['[SOS]']

        self.backbone.resize_token_embeddings(0)
        self.backbone.resize_token_embeddings(len(tkn2id))
        try:
            self.backbone.transformer.wte.padding_idx = tkn2id['[PAD]']
        except:
            self.backbone.model.embed_tokens.padding_idx = tkn2id['[PAD]']

        assert classifierType in ['CNN', 'RNN', 'BPNN', 'FFN', 'Linear']
        if classifierType=='CNN':
            self.textCNN = TextCNN(self.backbone.config.hidden_size, filterSize=self.backbone.config.hidden_size//4, contextSizeList=[1,5,25], reduction=reduction, ln=norm)
            self.classifier = nn.Sequential(
                                nn.Dropout(dropout),
                                nn.Linear(self.backbone.config.hidden_size//4 * 3 * (2 if 'ensemble' in reduction else 1), fcSize),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(fcSize, classNum)
                            )
            if norm: self.classifier.insert(2, nn.BatchNorm1d(fcSize))
        elif classifierType=='RNN':
            self.textRNN = TextLSTM(self.backbone.config.hidden_size, hiddenSize=self.backbone.config.hidden_size//2, reduction=reduction, ln=norm)
            self.classifier = nn.Sequential(
                                nn.Dropout(dropout),
                                nn.Linear(self.backbone.config.hidden_size//2 * 2 * (2 if 'ensemble' in reduction else 1), fcSize),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(fcSize, classNum)
                            )
            if norm: self.classifier.insert(2, nn.BatchNorm1d(fcSize))
        elif classifierType=='BPNN':
            self.classifier = nn.Sequential(
                                nn.Dropout(dropout),
                                nn.Linear(self.backbone.config.hidden_size*len(use_fea_list), fcSize),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(fcSize, classNum)
                            )
            if norm: self.classifier.insert(2, nn.BatchNorm1d(fcSize))
        elif classifierType=='FFN':
            self.ffn1 = nn.Sequential(
                            nn.Dropout(dropout),
                            nn.Linear(self.backbone.config.hidden_size*len(use_fea_list), fcSize),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(fcSize,self.backbone.config.hidden_size*len(use_fea_list))
                        )
            if norm: self.ffn1.insert(2, nn.BatchNorm1d(fcSize))
            self.ffn2 = nn.Sequential(
                            nn.Dropout(dropout),
                            nn.Linear(self.backbone.config.hidden_size*len(use_fea_list), fcSize),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(fcSize,self.backbone.config.hidden_size*len(use_fea_list))
                        )
            if norm: 
                self.ffn2.insert(2, nn.BatchNorm1d(fcSize))
                self.ffn2.insert(0, nn.BatchNorm1d(self.backbone.config.hidden_size*len(use_fea_list)))
            self.classifier = nn.Sequential(
                                nn.Dropout(dropout),
                                nn.Linear(self.backbone.config.hidden_size*len(use_fea_list), classNum)
                            )
            if norm:
                self.classifier.insert(0, nn.BatchNorm1d(self.backbone.config.hidden_size*len(use_fea_list)))
        else:
            self.classifier = nn.Sequential(
                                    nn.Dropout(dropout),
                                    nn.Linear(self.backbone.config.hidden_size*len(use_fea_list), classNum)
                                )
            if norm:
                self.classifier.insert(0, nn.BatchNorm1d(fcSize))

        self.classifierType = classifierType
        self.use_fea_list = use_fea_list

        # self.fcLinear = nn.Linear(self.backbone.config.hidden_size*2, 1)
    def forward(self, data):
        batch = data['batch']
        try:
            out = self.backbone.transformer(**({k:batch[k] for k in batch if k!='labels'})).last_hidden_state # => B × L × C
        except:
            out = self.backbone.model(**({k:batch[k] for k in batch if k!='labels'})).last_hidden_state # => B × L × C

        if self.classifierType=='CNN':
            out = self.textCNN(out, batch['attention_mask'])
        elif self.classifierType=='RNN':
            out = self.textRNN(out, batch['attention_mask'])
        else:
            tmp = []
            if 'sos' in self.use_fea_list:
                tmp.append( out[:,0] )
            if 'max' in self.use_fea_list:
                out_max,_ = torch.max(out, dim=1) # => B × C
                tmp.append( out_max )
            if 'mean' in self.use_fea_list:
                out_mean = (out*batch['attention_mask'].unsqueeze(dim=-1)).sum(dim=1) / batch['attention_mask'].sum(dim=1).unsqueeze(dim=-1)
                tmp.append(out_mean)
            out = torch.cat(tmp, dim=1)
            if self.classifierType=='FFN':
                out2 = out+self.ffn1(out)
                out3 = out2+self.ffn2(out2)
                out = out3

        return {'y_logit':self.classifier(out)}