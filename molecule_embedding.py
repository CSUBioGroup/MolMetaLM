import argparse,os,pickle,datetime

parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--output_file')
parser.add_argument('--pretrainPath', default='', type=str)

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--prefetch_factor', type=int, default=16)

parser.add_argument('--ddp', default='false', type=str)

args = parser.parse_args()

from utils import *
from DL_ClassifierModel import *
import rdkit

if __name__=='__main__':
    ddp = args.ddp=='true'
    print(os.system('hostname'))
    if ddp:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank) # 设定cuda的默认GPU，每个rank不同
        print('local_rank:', args.local_rank)
        torch.distributed.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=36000)) # , world_size=args.dataLoadNumWorkers


    collater = HuggingfaceNoisingCollateFunc_final(bertDir='./bertModel/Tokenizer_final', seqMaxLen=512, k=64)
    tkn2id = collater.tokenizer.get_vocab()
    finetuneCollater = InferenceCollateFunc(bertDir='./bertModel/Tokenizer_final', seqMaxLen=512, \
                                            prompt="[SMILES]")

    config = AutoConfig.from_pretrained("./bertModel/cus-llama2-base", trust_remote_code=True, use_flash_attention_2=True)
    backbone = MolMetaLM(config, tkn2id, maxGenLen=512).cuda()
    backbone.alwaysTrain = False
    
    if ddp:
        backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True) #

    model = HuggingfaceSeq2SeqLanguageModel(backbone, collateFunc=finetuneCollater, AMP=True, DDP=ddp)

    if len(args.pretrainPath)>0:
        model.load(args.pretrainPath)
    else:
        model.model.backbone = AutoModelForCausalLM.from_pretrained('wudejian789/MolMetaLM-base').cuda()
        
    model.to_eval_mode()
    finetuneCollater.padding = 'longest'

    dataset = DataClass_normal(args.input_file, randomSMILES=False)
    
    if ddp:
        validSampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        validSampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=int(args.batch_size), collate_fn=finetuneCollater,
                            pin_memory=True, num_workers=int(args.num_workers), 
                            prefetch_factor=int(args.prefetch_factor), sampler=validSampler)

    print('predicting...')
    if args.output_file.endswith('.pkl'):
        res = []
        for batch in tqdm(dataloader):
            batch = dict_to_device(batch, 'cuda')
            with torch.no_grad():
                if ddp:
                    vec = model.model.module.backbone(**(batch['batch']), 
                                                      output_hidden_states=True).hidden_states[-1].max(axis=1)[0].squeeze().detach().cpu().data.numpy()
                else:
                    vec = model.model.backbone(**(batch['batch']), 
                                               output_hidden_states=True).hidden_states[-1].max(axis=1)[0].squeeze().detach().cpu().data.numpy()
                res.append(vec)
        res = np.vstack(res).astype('float32') # N, 768
        
        with open(args.output_file, 'wb') as f:
            pickle.dump(res, f)

    elif args.output_file.endswith('.mem'):
        bs = int(args.batch_size)
        if ddp:
            fp = np.memmap(f"{args.output_file}_{args.local_rank}", dtype='float32', mode='w+', shape=(len(dataset),768))
        else:
            fp = np.memmap(args.output_file, dtype='float32', mode='w+', shape=(len(dataset),768))
        for i,batch in enumerate(tqdm(dataloader)):
            batch = dict_to_device(batch, 'cuda')
            with torch.no_grad():
                if ddp:
                    vec = model.model.module.backbone(**(batch['batch']), 
                                                      output_hidden_states=True).hidden_states[-1].max(axis=1)[0].squeeze().detach().cpu().data.numpy()
                else:
                    vec = model.model.backbone(**(batch['batch']), 
                                               output_hidden_states=True).hidden_states[-1].max(axis=1)[0].squeeze().detach().cpu().data.numpy()
            fp[i*bs:(i+1)*bs] = vec
            # fp.flush()
        fp.flush()


# python UniMolLM_fpr.py  --input_file ../2024DTI_NIPS/fpr_cache/synsmiles_test.txt --output_file ../2024DTI_NIPS/fpr_cache/synsmiles_test_unimollm_vec.mem  --batch_size 64