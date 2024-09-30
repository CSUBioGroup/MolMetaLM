import argparse,torch,pickle,traceback,gc
import torch.distributed
from utils import *
from DL_ClassifierModel import *
from ImageMol_splitter import *

def get_agbt_dataset_split_by_scaffold(name, SEED=9527):
    if name in ['LD50','LC50','IGC50','LC50DM']:
        trainDS,testDS = Toxicity(f'./datasets/AGBT/toxicity/{name}',type='training'),Toxicity(f'./datasets/AGBT/toxicity/{name}',type='prediction')
        totalDS = torch.utils.data.ConcatDataset([trainDS,testDS])
        trainIdx,validIdx = train_test_split(range(len(trainDS)), test_size=0.1, random_state=SEED)
        trainDS,validDS = torch.utils.data.Subset(trainDS,trainIdx),torch.utils.data.Subset(trainDS,validIdx)
    elif name=='LogP':
        trainDS,testDS = PartitionCoefficient('./datasets/AGBT/logP-logS/logP/training-8199'),PartitionCoefficient('./datasets/AGBT/logP-logS/logP/FDA')
        totalDS = torch.utils.data.ConcatDataset([trainDS,testDS])
        trainIdx,validIdx = train_test_split(range(len(trainDS)), test_size=0.1, random_state=SEED)
        trainDS,validDS = torch.utils.data.Subset(trainDS,trainIdx),torch.utils.data.Subset(trainDS,validIdx)
    elif name in ['FreeSolv']:
        totalDS = FreeSolv('./datasets/AGBT/FreeSolv')
        trainIdx,testIdx = train_test_split(range(len(totalDS)), test_size=0.2, random_state=SEED)
        validIdx,testIdx = train_test_split(testIdx, test_size=0.5, random_state=SEED)
        trainDS,testDS,validDS = torch.utils.data.Subset(totalDS, trainIdx),torch.utils.data.Subset(totalDS, validIdx),torch.utils.data.Subset(totalDS, testIdx)
    elif name=='Lipophilicity':
        totalDS = Lipophilicity('./datasets/AGBT/Lipophilicity')
        trainIdx,testIdx = train_test_split(range(len(totalDS)), test_size=0.2, random_state=SEED)
        validIdx,testIdx = train_test_split(testIdx, test_size=0.5, random_state=SEED)
        trainDS,testDS,validDS = torch.utils.data.Subset(totalDS, trainIdx),torch.utils.data.Subset(totalDS, validIdx),torch.utils.data.Subset(totalDS, testIdx)

    tmp = np.array([i['y'] for i in trainDS])
    mean,std = tmp.mean(axis=0),tmp.std(axis=0)

    return totalDS,trainDS,validDS,testDS,(mean,std)

def get_gpcr_dataset_split_by_scaffold(name):
    totalDS = MPP(f'./datasets/GPCR/{name}/processed/{name}_processed_ac.csv')
    trainIdx,validIdx,testIdx = scaffold_split_balanced_train_val_test(range(len(totalDS)), totalDS.smilesList)
    
    tmp = np.array(totalDS.Y, dtype=np.float32)[trainIdx]
    mean,std = tmp.mean(axis=0, keepdims=True),tmp.std(axis=0, keepdims=True)
    
    trainDS,validDS,testDS = torch.utils.data.Subset(totalDS, trainIdx),torch.utils.data.Subset(totalDS, validIdx),torch.utils.data.Subset(totalDS, testIdx)

    return totalDS,trainDS,validDS,testDS,(mean,std)

def get_dc_molnet_dataset_split_by_scaffold(name):
    mean,std = 0,1

    trainDS = MoleculeNetDC(name, type='train')
    validDS = MoleculeNetDC(name, type='valid')
    testDS = MoleculeNetDC(name, type='test')

    if name in ['ESOL','FreeSolv','Lipophilicity','QM7','QM8','QM9']:
        tmp = np.array(trainDS.Y)
        mean,std = tmp.mean(axis=0),tmp.std(axis=0)

    totalDS = torch.utils.data.ConcatDataset([trainDS,validDS,testDS])

    return totalDS,trainDS,validDS,testDS,(mean,std)

def get_unimol_molnet_dataset_split_by_scaffold(name):
    mean,std = 0,1
    
    trainDS = UniMolMoleculeNetDS(name, type='train')
    validDS = UniMolMoleculeNetDS(name, type='valid')
    testDS = UniMolMoleculeNetDS(name, type='test')
    
    if name in ['ESOL','FreeSolv','Lipophilicity','QM7','QM8','QM9']:
        tmp = np.array(trainDS.Y)
        mean,std = tmp.mean(axis=0, keepdims=True),tmp.std(axis=0, keepdims=True)
    
    totalDS = torch.utils.data.ConcatDataset([trainDS,validDS,testDS])
    
    return totalDS,trainDS,validDS,testDS,(mean,std)

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, help='ds1;ds2;...;dsn;')
parser.add_argument('--taskType', default='regression', choices=['regression','classification'], type=str)
parser.add_argument('--modelType', default='llama2-small', type=str, choices=['llama2-small','llama2-base'])
parser.add_argument('--pretrainPath', default='', type=str)
parser.add_argument('--savePath', default='./saved_models/', type=str)
parser.add_argument('--summaryCSV', default='', type=str)
parser.add_argument('--promptType', default='5', choices=['4','5'])
parser.add_argument('--SEED', default=9527, type=int)
parser.add_argument('--bsList', type=str, default='64;128;256') # 3
parser.add_argument('--lrList', type=str, default='3e-5;1e-4;3e-4;1e-3') # 4
parser.add_argument('--wdList', type=str, default='1e-1;1e-2;1e-3;0.0') # 4
parser.add_argument('--dpList', type=str, default='0.0;0.1;0.2') # 3
parser.add_argument('--wrList', type=str, default='0.1;0.0') # 2
parser.add_argument('--classifierType', type=str, default='BPNN')

parser.add_argument('--sList', type=str, default='40;2024;1234;8791;7268;8888;6666;618') # 2
parser.add_argument('--metrics', type=str, default='')
parser.add_argument('--reverse', type=bool, default=False)
args = parser.parse_args()

if __name__ == '__main__':
    step = -1 if args.reverse else 1
    if args.reverse:
        args.summaryCSV = ""
    if len(args.summaryCSV)>0:
        f = open(args.summaryCSV, 'w')
        f.write('datasets, models, pretrained, promptType, seed, bs, lr, wd, dp, wr, RMSE/MAE/R2, MaAUROC/MaAUPRC/ACC\n')
    else:
        f = None

    bsList = [int(i) for i in args.bsList.split(';')]
    lrList = [float(i) for i in args.lrList.split(';')]
    wdList = [float(i) for i in args.wdList.split(';')]
    sList = [int(i) for i in args.sList.split(';')]
    dpList = [float(i) for i in args.dpList.split(';')]
    wrList = [float(i) for i in args.wrList.split(';')]
    metrics = args.metrics

    for dataset in args.datasets.split(';'):
        torch.cuda.empty_cache()
        print(f'Finetuning dataset {dataset}...')
        if dataset.startswith('UniMol_'):
            totalDS,trainDS,validDS,testDS,(mean,std) = get_unimol_molnet_dataset_split_by_scaffold(dataset[dataset.find('_')+1:])
        elif dataset.startswith('GPCR_'):
            totalDS,trainDS,validDS,testDS,(mean,std) = get_gpcr_dataset_split_by_scaffold(dataset[dataset.find('_')+1:])
        elif dataset.startswith('AGBT_'):
            totalDS,trainDS,validDS,testDS,(mean,std) = get_agbt_dataset_split_by_scaffold(dataset[dataset.find('_')+1:], args.SEED)
        else:
            totalDS,trainDS,validDS,testDS,(mean,std) = get_dc_molnet_dataset_split_by_scaffold(dataset)

        try:
            try:
                allSmiles = set(trainDS.smilesList+validDS.smilesList+testDS.smilesList)
            except:
                allSmiles = set(totalDS.smilesList)
        except:
            allSmiles = set([i['smiles'] for i in totalDS])
        args.promptType = int(args.promptType)
        if args.promptType<4 or args.promptType==5:
            tmp = [len(i) for i in allSmiles]
        else:
            tmp = [len(i)+len([i.GetSymbol() for i in Chem.MolFromSmiles(i).GetAtoms() if i.GetSymbol()!='H'])*3 for i in allSmiles]
        # seqMaxLen = int(np.quantile(tmp, 0.95))+10
        seqMaxLen = min(np.max(tmp)+2, 384)

        print('Using seqMaxLen:', seqMaxLen)
        if args.promptType==4:
            finetuneCollater = FinetuneCollateFunc_final(bertDir='./bertModel/Tokenizer_final', seqMaxLen=seqMaxLen, prompt="[sPPM] [SMILES] [SEP] [CUSPRO] [VALUE] ; [SEP] [SPM_DDD] [(DDD)STRUCTURE] [SEP] [SPM_AAD] [(AAD)STRUCTURE] [SPM_XYZ] [(XYZ)STRUCTURE] [SOS]", 
                                                         randomSMILES=True, useStructure=True)
        elif args.promptType==5:
            finetuneCollater = FinetuneCollateFunc_final(bertDir='./bertModel/Tokenizer_final', seqMaxLen=seqMaxLen, prompt="[SMILES]", randomSMILES=True)
        
        if args.taskType=='regression':
            finetuneCollater.normVec = (mean,std)

        tkn2id = finetuneCollater.tokenizer.get_vocab()

        bestParams = []
        epochs = 256
        earlyStop = 64
        warmupRatio = 0
        
        config = AutoConfig.from_pretrained(f"./bertModel/cus-{args.modelType}", trust_remote_code=True)
        if args.taskType=='regression':
            report = ['RMSE', 'MAE', 'R2']
            if len(metrics)==0: 
                metrics = 'RMSE' if not args.datasets.startswith('AGBT_') else 'R2'
            isHigherBetter = False if not args.datasets.startswith('AGBT_') else True
        elif args.taskType=='classification':
            report = ['MaAUC','MaAUPR','ACC']
            if args.datasets.startswith('UniMol_'):
                report = ['ValidMaAUC', 'ValidMaAUPR', 'ValidACC']
            if len(metrics)==0:
                metrics = 'MaAUC'
                if args.datasets.startswith('UniMol_'):
                    metrics = 'ValidMaAUC'
            isHigherBetter = True

        m = None
        for bs in bsList[::step]:
            for lr in lrList[::step]:
                for wd in wdList[::step]:
                    for dp in dpList[::step]:
                        for wr in wrList[::step]:
                            torch.cuda.empty_cache()

                            set_seed(args.SEED)
                            backbone = MolMetaLM_finetune_Alpha(config, tkn2id, classNum=len(totalDS[0]['y']), classifierType=args.classifierType, dropout=dp).cuda()
                            stepsPerEpoch = len(trainDS)//bs
                            backbone.alwaysTrain = True
                            
                            if len(args.pretrainPath)>0:
                                parameters = torch.load(args.pretrainPath)
                                backbone.load_state_dict(parameters['model'], strict=False)
                                print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc'] if parameters['bestMtc'] is not None else 0.00))

                            backbone.backbone.resize_token_embeddings(len(finetuneCollater.tokenizer.get_vocab()))
                            
                            pretrain = ['backbone'] # 'encoder' 
                            no_decay = ['bias', 'LayerNorm.weight'] # 
                            optimizer_grouped_parameters = [
                                    {'params': [p for n, p in backbone.named_parameters() if (not any(nd in n for nd in no_decay)) and (not any(nd in n for nd in pretrain)) ],
                                     'weight_decay': wd, 'lr': lr},
                                    {'params': [p for n, p in backbone.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in pretrain)], 
                                     'weight_decay': 0.0, 'lr': lr/3}, # lr/3 0.0
                                    {'params': [p for n, p in backbone.named_parameters() if (not any(nd in n for nd in no_decay)) and any(nd in n for nd in pretrain)],
                                     'weight_deca y': wd, 'lr':lr/3}, # lr / 3
                                    {'params': [p for n, p in backbone.named_parameters() if any(nd in n for nd in no_decay) and (not any(nd in n for nd in pretrain))], 
                                     'weight_decay': 0.0, 'lr':lr}, # 0.0
                                ]
                            
                            optimizer =  torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=wd)

                            if args.taskType=='regression':
                                model = HuggingfaceSeq2SeqLanguageModel_ForRegression(backbone, optimizer=optimizer, collateFunc=finetuneCollater, AMP=True)
                            elif args.taskType=='classification':
                                model = HuggingfaceSeq2SeqLanguageModel_ForClassification(backbone, optimizer=optimizer, collateFunc=finetuneCollater, AMP=True, multilabel=True)
                                if args.datasets.startswith('UniMol_'):
                                    model.criterion = ValidBCELoss()
                            if metrics=='MAE':
                                model.criterion = nn.SmoothL1Loss()

                            weight = [i for i in os.listdir(args.savePath) if i.startswith(f'finetune_{dataset}_{args.modelType}_{args.classifierType}_bs{bs}_lr{lr}_wd{wd}_dp{dp}_wr{wr}_s{args.SEED}_p{args.promptType}_')]
                            assert len(weight)<=1
                            try:
                                if len(weight)==0:
                                    res = model.train(trainDataSet=trainDS, validDataSet=validDS, batchSize=bs, doEvalTrain=False,
                                                      saveSteps=-1, maxSteps=stepsPerEpoch*epochs, evalSteps=stepsPerEpoch, earlyStop=earlyStop, 
                                                      metrics=metrics, report=report, isHigherBetter=isHigherBetter,
                                                      savePath=f"{args.savePath}/finetune_{dataset}_{args.modelType}_{args.classifierType}_bs{bs}_lr{lr}_wd{wd}_dp{dp}_wr{wr}_s{args.SEED}_p{args.promptType}", 
                                                      dataLoadNumWorkers=8, pinMemory=True, prefetchFactor=16,
                                                      warmupSteps=int(wr*stepsPerEpoch), SEED=args.SEED)
                                    with torch.no_grad():
                                        res = model.calculate_metrics_by_iterator(DataLoader(testDS, batch_size=bs, collate_fn=finetuneCollater, 
                                                                                             num_workers=8, pin_memory=True, prefetch_factor=16), 
                                                                                  Metrictor(), ignoreIdx=-100, report=report,
                                                                                  TTA_num=64, TTA_obj=metrics, isHigherBetter=isHigherBetter, tol=1.0)
                                else:
                                    weight = weight[0]
                                    print(f'Find a already trained model in {os.path.join(args.savePath,weight)}, loading it for prediction directly!')
                                    model.load(os.path.join(args.savePath, weight))
                                    model.to_eval_mode()
                                    print(f'[Total Valid]',end='')
                                    with torch.no_grad():
                                        res = model.calculate_metrics_by_iterator(DataLoader(testDS, batch_size=bs, collate_fn=finetuneCollater, 
                                                                                             num_workers=8, pin_memory=True, prefetch_factor=16), 
                                                                                  Metrictor(), ignoreIdx=-100, report=report,
                                                                                  TTA_num=64, TTA_obj=metrics, isHigherBetter=isHigherBetter, tol=1.0)
                                    print(res)
                                for k in list(res.keys()):
                                    res[k.replace("Valid","")] = res[k]
                                if (m is None) or (isHigherBetter and res[metrics]>m) or (not isHigherBetter and res[metrics]<m):
                                    m = res[metrics]
                                    bestRES = res
                                    bestParams = [bs,lr,wd,dp,wr]
                            except:
                                print(f'problems while training finetune_{dataset}_{args.modelType}_{args.classifierType}_bs{bs}_lr{lr}_wd{wd}_dp{dp}_wr{wr}_s{args.SEED}_p{args.promptType}...')
                                traceback.print_exc()

                            del backbone,model
                            gc.collect()

        if f is not None:
            bs,lr,wd,dp,wr = bestParams
            res = bestRES
            if args.taskType=='regression':
                RMSE = float(res['RMSE']) # float(np.sqrt(res['RMSE']**2 *std*std))
                MAE = float(res['MAE']) # float(res['MAE'] *std)
                R2 = float(res['R2']) # float(res['R2'])

                f.write(f'{dataset}, {args.modelType}, {args.pretrainPath}, {args.promptType}, {args.SEED}, {bs}, {lr}, {wd}, {dp}, {wr}, {RMSE:.3f}/{MAE:.3f}/{R2:.3f}, -\n')
            elif args.taskType=='classification':
                AUC = res['MaAUC']
                AUPR = res['MaAUPR']
                ACC = res['ACC']

                f.write(f'{dataset}, {args.modelType}, {args.pretrainPath}, {args.promptType}, {args.SEED}, {bs}, {lr}, {wd}, {dp}, {wr}, -, {AUC:.3f}/{AUPR:.3f}/{ACC:.3f}\n')

        # multi seed test
        for seed in [args.SEED] + sList[::step]:
            torch.cuda.empty_cache()
            if dataset.startswith('AGBT_'):
                totalDS,trainDS,validDS,testDS,(mean,std) = get_agbt_dataset_split_by_scaffold(dataset[dataset.find('_')+1:], seed)
                finetuneCollater.normVec = (mean,std)

            bs,lr,wd,dp,wr = bestParams
            set_seed(seed)
            backbone = MolMetaLM_finetune_Alpha(config, tkn2id, classNum=len(totalDS[0]['y']), classifierType=args.classifierType, dropout=dp).cuda()
            stepsPerEpoch = len(trainDS)//bs
            backbone.alwaysTrain = True
            
            if len(args.pretrainPath)>0:
                parameters = torch.load(args.pretrainPath)
                backbone.load_state_dict(parameters['model'], strict=False)
                print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc'] if parameters['bestMtc'] is not None else 0.00))

            backbone.backbone.resize_token_embeddings(len(finetuneCollater.tokenizer.get_vocab()))
            
            pretrain = ['backbone'] # 'encoder' 
            no_decay = ['bias', 'LayerNorm.weight'] # 
            optimizer_grouped_parameters = [
                    {'params': [p for n, p in backbone.named_parameters() if (not any(nd in n for nd in no_decay)) and (not any(nd in n for nd in pretrain)) ],
                     'weight_decay': wd, 'lr': lr},
                    {'params': [p for n, p in backbone.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in pretrain)], 
                     'weight_decay': 0.0, 'lr': lr/3}, # lr/3 0.0
                    {'params': [p for n, p in backbone.named_parameters() if (not any(nd in n for nd in no_decay)) and any(nd in n for nd in pretrain)],
                     'weight_deca y': wd, 'lr':lr/3}, # lr / 3
                    {'params': [p for n, p in backbone.named_parameters() if any(nd in n for nd in no_decay) and (not any(nd in n for nd in pretrain))], 
                     'weight_decay': 0.0, 'lr':lr}, # 0.0
                ]

            optimizer =  torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=wd)

            if args.taskType=='regression':
                model = HuggingfaceSeq2SeqLanguageModel_ForRegression(backbone, optimizer=optimizer, collateFunc=finetuneCollater, AMP=True)
            elif args.taskType=='classification':
                model = HuggingfaceSeq2SeqLanguageModel_ForClassification(backbone, optimizer=optimizer, collateFunc=finetuneCollater, AMP=True, multilabel=True)
                if args.datasets.startswith('UniMol_'):
                    model.criterion = ValidBCELoss()
            if metrics=='MAE':
                model.criterion = nn.SmoothL1Loss()

            weight = [i for i in os.listdir(args.savePath) if i.startswith(f'finetune_{dataset}_{args.modelType}_{args.classifierType}_bs{bs}_lr{lr}_wd{wd}_dp{dp}_wr{wr}_s{seed}_p{args.promptType}_')]
            assert len(weight)<=1        
            try:
                if len(weight)==0:
                    res = model.train(trainDataSet=trainDS, validDataSet=validDS, batchSize=bs, doEvalTrain=False,
                                      saveSteps=-1, maxSteps=stepsPerEpoch*epochs, evalSteps=stepsPerEpoch, earlyStop=earlyStop, 
                                      metrics=metrics, report=report, isHigherBetter=isHigherBetter,
                                      savePath=f"{args.savePath}/finetune_{dataset}_{args.modelType}_{args.classifierType}_bs{bs}_lr{lr}_wd{wd}_dp{dp}_wr{wr}_s{seed}_p{args.promptType}", 
                                      dataLoadNumWorkers=8, pinMemory=True, prefetchFactor=16,
                                      warmupSteps=int(wr*stepsPerEpoch), SEED=seed)
                    with torch.no_grad():
                        res = model.calculate_metrics_by_iterator(DataLoader(testDS, batch_size=bs, collate_fn=finetuneCollater, 
                                                                             num_workers=8, pin_memory=True, prefetch_factor=16), 
                                                                  Metrictor(), ignoreIdx=-100, report=report,
                                                                  TTA_num=64, TTA_obj=metrics, isHigherBetter=isHigherBetter, tol=1.0)
                else:
                    weight = weight[0]
                    print(f'Find a already trained model in {os.path.join(args.savePath,weight)}, loading it for prediction directly!')
                    model.load(os.path.join(args.savePath, weight))
                    model.to_eval_mode()
                    print(f'[Total Valid]',end='')
                    with torch.no_grad():
                        res = model.calculate_metrics_by_iterator(DataLoader(testDS, batch_size=bs, collate_fn=finetuneCollater, 
                                                                             num_workers=8, pin_memory=True, prefetch_factor=16), 
                                                                  Metrictor(), ignoreIdx=-100, report=report,
                                                                  TTA_num=64, TTA_obj=metrics, isHigherBetter=isHigherBetter, tol=1.0)
                    print(res)
                for k in list(res.keys()):
                    res[k.replace("Valid","")] = res[k]
                if f is not None:
                    if args.taskType=='regression':
                        RMSE = float(res['RMSE']) # float(np.sqrt(res['RMSE']**2 *std*std))
                        MAE = float(res['MAE']) # float(res['MAE'] *std)
                        R2 = float(res['R2']) # float(res['R2'])

                        f.write(f'{dataset}, {args.modelType}, {args.pretrainPath}, {args.promptType}, {seed}, {bs}, {lr}, {wd}, {dp}, {wr}, {RMSE:.3f}/{MAE:.3f}/{R2:.3f}, -\n')
                    elif args.taskType=='classification':
                        AUC = res['MaAUC']
                        AUPR = res['MaAUPR']
                        ACC = res['ACC']

                        f.write(f'{dataset}, {args.modelType}, {args.pretrainPath}, {args.promptType}, {seed}, {bs}, {lr}, {wd}, {dp}, {wr}, -, {AUC:.3f}/{AUPR:.3f}/{ACC:.3f}\n')
            
            except:
                print(f'problems while training finetune_{dataset}_{args.modelType}_{args.classifierType}_bs{bs}_lr{lr}_wd{wd}_dp{dp}_wr{wr}_s{seed}_p{args.promptType}...')
                traceback.print_exc()

            del backbone,model
            gc.collect()
    f.close()
