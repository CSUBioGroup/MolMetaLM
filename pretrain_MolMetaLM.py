import argparse,torch,pickle
import torch.distributed
from utils import *
from DL_ClassifierModel import *
import datetime

# python -u -m torch.distributed.launch --nproc_per_node=4 train_main.py --batchSize 8 --maxSteps 200000 --evalSteps 3000 --earlyStop 16 --lr 0.00002 --warmupSteps 3000 --mode 1

# torch.backends.cudnn.benchmark = True
SEED = 9527

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', default=64, type=int)
parser.add_argument('--seqMaxLen', default=512, type=int)
parser.add_argument('--maxSteps', default=1000000, type=int)
parser.add_argument('--evalSteps', default=100, type=int)
parser.add_argument('--earlyStop', default=16, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weightDecay', default=0.001, type=float)
parser.add_argument('--warmupSteps', default=10000, type=float)
parser.add_argument('--lossType', default='CEL', type=str)
parser.add_argument('--restore', default=None, type=str)
parser.add_argument('--beginSteps', default=-1, type=int)
parser.add_argument('--dataset', default='pubchem', type=str)
parser.add_argument('--ddp', default='false', type=str)
parser.add_argument('--model_size', default='small', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    ddp = args.ddp=='true'
    print(os.system('hostname'))
    if ddp:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank) # 设定cuda的默认GPU，每个rank不同
        print('local_rank:', args.local_rank)
        torch.distributed.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=36000)) # , world_size=args.dataLoadNumWorkers

    collater = HuggingfaceNoisingCollateFunc_final(bertDir='./bertModel/Tokenizer_final', seqMaxLen=args.seqMaxLen, k=64)
    tkn2id = collater.tokenizer.get_vocab()

    config = AutoConfig.from_pretrained(f"./bertModel/cus-llama2-{args.model_size}", trust_remote_code=True)
    # config.position_encoding_2d = False
    backbone = MolMetaLM(config, tkn2id, maxGenLen=args.seqMaxLen).cuda()
    backbone.alwaysTrain = True

    if args.restore is not None:
        print('Using restored weight...')
        parameters = torch.load(args.restore, map_location=f"cuda:{args.local_rank}" if ddp else "cuda")
        backbone.load_state_dict(parameters['model'], strict=False)
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc'] if parameters['bestMtc'] is not None else 0.00))
        backbone.backbone.resize_token_embeddings(len(collater.tokenizer.get_vocab()))

    if ddp:
        backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True) #

    pretrain = []
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in backbone.named_parameters() if (not any(nd in n for nd in no_decay)) and (not any(nd in n for nd in pretrain)) ],
             'weight_decay': args.weightDecay, 'lr': args.lr},
            {'params': [p for n, p in backbone.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in pretrain)], 
             'weight_decay': 0.0, 'lr': args.lr/3},
            {'params': [p for n, p in backbone.named_parameters() if (not any(nd in n for nd in no_decay)) and any(nd in n for nd in pretrain)],
             'weight_deca y': args.weightDecay, 'lr':args.lr/3},
            {'params': [p for n, p in backbone.named_parameters() if any(nd in n for nd in no_decay) and (not any(nd in n for nd in pretrain))], 
             'weight_decay': 0.0, 'lr':args.lr},
        ]

    optimizer =  torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weightDecay, eps=1e-6)

    if args.lossType == 'FCEL':
        criterion = FocalCrossEntropyLoss(ignoreIdx=tkn2id['[PAD]'], gama=2, smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=tkn2id['[PAD]'], label_smoothing=0.1) # 0.1
    model = HuggingfaceSeq2SeqLanguageModel(backbone, collateFunc=collater, AMP=True, DDP=ddp, 
                                            criterion=criterion,
                                            optimizer=optimizer)
    
    if args.dataset=='pubchem':
        if not os.path.exists('./pubchem.pkl'):
            dataset = PubChemDB('./datasets/PubChem/CID-SMILES')
            with open('pubchem.pkl', 'wb') as f:
                pickle.dump(dataset, f)
        else:
            with open('pubchem.pkl', 'rb') as f:
                dataset = pickle.load(f)
    else:
        dataset = DataClass_normal(args.dataset)

    totalDS = torch.utils.data.ConcatDataset([dataset])

    if len(totalDS)>=100000000:
        trainIdx,validIdx = train_test_split(range(len(totalDS)), test_size=0.0005, random_state=9527) # 0.0005
    elif len(totalDS)>=10000000:
        trainIdx,validIdx = train_test_split(range(len(totalDS)), test_size=0.005, random_state=9527) # 0.0005
    elif len(totalDS)>=1000000:
        trainIdx,validIdx = train_test_split(range(len(totalDS)), test_size=0.05, random_state=9527) # 0.0005
    else:
        trainIdx,validIdx = train_test_split(range(len(totalDS)), test_size=0.1, random_state=9527) # 0.0005
    
    trainDS,validDS = torch.utils.data.Subset(totalDS, trainIdx),torch.utils.data.Subset(totalDS, validIdx)
    print(len(trainDS), len(validDS))

    model.train(trainDataSet=trainDS, validDataSet=validDS, batchSize=args.batchSize, beginSteps=args.beginSteps,
                maxSteps=args.maxSteps, saveSteps=args.evalSteps, evalSteps=args.evalSteps, earlyStop=args.earlyStop, 
                metrics="LOSS", report=['LOSS'], isHigherBetter=False, 
                savePath=f"./saved_models/MolMetaLM_{args.model_size}", dataLoadNumWorkers=4, pinMemory=True, ignoreIdx=tkn2id['[PAD]'], 
                warmupSteps=args.warmupSteps, SEED=SEED, prefetchFactor=16, tensorboard=False)

    backbone.alwaysTrain = False

