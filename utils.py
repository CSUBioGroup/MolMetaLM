import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
import torch,random,os,jieba,re
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from itertools import permutations
from rdkit.Chem import AllChem
from rdkit import Chem
import deepchem
from deepchem.models.graph_models import GraphConvModel
from deepchem.feat import graph_features

from tools import *

import logging,pickle,gc
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from typing import Union

# --- UTILITY FUNCTIONS ---
def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)

def sequence_to_string(seq: np.ndarray) -> str:
    return ''.join([chr(c) for c in seq])

def pack_sequences(seqs: Union[np.ndarray, list]) -> (np.ndarray, np.ndarray):
    values = np.concatenate(seqs, axis=0)
    offsets = np.cumsum([len(s) for s in tqdm(seqs)])
    return values, offsets

def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    off1 = offsets[index]
    if index > 0:
        off0 = offsets[index - 1]
    elif index == 0:
        off0 = 0
    else:
        raise ValueError(index)
    return values[off0:off1]

class DataClass_normal(Dataset):
    def __init__(self, path, randomSMILES=True):
        self.path = path
        print('loading...')
        with open(path, 'r') as f:
            smiles = [string_to_sequence(smi.strip()) for smi in tqdm(f.readlines())]
        print('packing...')
        self.len = len(smiles)
        self.smiles_v,self.smiles_o = pack_sequences(smiles)
        self.randomSMILES = randomSMILES
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        smiles = sequence_to_string(unpack_sequence(self.smiles_v, self.smiles_o, index))

        mol = Chem.MolFromSmiles(smiles) if len(smiles)>0 else None
        if self.randomSMILES:
            try:
                smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False, isomericSmiles=True) if mol is not None else ""
                mol = Chem.MolFromSmiles(smi) if len(smi)>0 else None
            except:
                smi = smiles
                gc.collect()
        else:
            smi = smiles
        return {'smiles':smi, \
                'mol':mol}

class PubChemDB(Dataset):
    def __init__(self, path, samples=None):
        self.path = path
        print('Loading SMILES from PubChemDB...')
        with open(path, 'r') as f:
            if isinstance(samples, int):
                lines = [i.strip().split() for i in tqdm(f.readlines()[:samples])]
            elif isinstance(samples, list):
                tmp = f.readlines()
                lines = [tmp[i].strip().split() for i in tqdm(samples)]
            else:
                lines = [i.strip().split() for i in tqdm(f.readlines())]
        print('string to sequencing...')
        self.cids,smiles = np.array([i[0] for i in lines], dtype=np.int32),[string_to_sequence(i[1]) for i in tqdm(lines)]
        print('packing...')
        self.smiles_v,self.smiles_o = pack_sequences(smiles)
    def __len__(self):
        return len(self.cids)
    def __getitem__(self, index):
        smiles = sequence_to_string(unpack_sequence(self.smiles_v, self.smiles_o, index))
        
        mol = Chem.MolFromSmiles(smiles) if len(smiles)>0 else None
        try:
            smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False, isomericSmiles=True) if mol is not None else ""
            mol = Chem.MolFromSmiles(smi) if len(smi)>0 else None
        except:
            smi = smiles
            gc.collect()
        return {'cid':self.cids[index],      \
                'smiles':smi, \
                'mol':mol}

# AGBT: LC50DM, Figure 2(b); IGC50, Figure 2(a); LD50, Table 1; LC50, Table 1. (toxicity prediction)
class Toxicity(Dataset):
    def __init__(self, path, type):
        assert type in ['training','prediction']
        self.path = path
        self.smilesList,self.molList,self.Y = [],[],[]
        
        validList = []
        for file in sorted(os.listdir(self.path+f'/{type}')):
            mol = Chem.SDMolSupplier(os.path.join(self.path+f'/{type}', file))[0]
            if mol is None:
                print(f'error in loading molecule with id {file}...')
                continue
            validList.append(file)
            canSMI = Chem.MolToSmiles(mol, canonical=True)
            self.smilesList.append(canSMI)
            self.molList.append(Chem.MolFromSmiles(canSMI))
        for file in validList:
            with open(os.path.join(self.path+f'/{type}_target', file.replace('sdf','exp'))) as f:
                exp = float(f.readlines()[0].strip())
            self.Y.append([exp])
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):
        return {'smiles':self.smilesList[index],
                'mol':self.molList[index],
                'y':self.Y[index]}

# AGBT: Table 1, FDA. (partition coefficient prediction)
class PartitionCoefficient(Dataset):
    def __init__(self, path):
        self.path = path
        self.smilesList,self.molList,self.Y = [],[],[]

        validList = []
        for file in tqdm(sorted(os.listdir(self.path+'/mols'))):
            mol = Chem.MolFromMol2File(os.path.join(self.path+'/mols', file))
            if mol is None:
                print(f'error in loading molecule with id {file}...')
                continue
            validList.append(file)
            canSMI = Chem.MolToSmiles(mol, canonical=True)
            self.smilesList.append(canSMI)
            self.molList.append(Chem.MolFromSmiles(canSMI))
        for file in tqdm(validList):
            with open(os.path.join(self.path+'/exps', file.replace('mol2','exp'))) as f:
                exp = float(f.readlines()[0].strip())
            self.Y.append([exp])
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):
        return {'smiles':self.smilesList[index],
                'mol':self.molList[index],
                'y':self.Y[index]}

# AGBT: Table 1, FreeSolv 
# (following MoleculeNet, set different random seeds and follow the same procedure ten times to obtain ten different data splitting)
class FreeSolv(Dataset):
    def __init__(self, path):
        self.path = path
        data = pd.read_csv(os.path.join(path, 'SAMPL.csv'))
        self.smilesList,self.molList,self.Y = [],[],[]
        for item in data.itertuples():
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(item.smiles), canonical=True)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f'error in loading molecule with id {item.iupac}')
                continue
            self.smilesList.append(smi)
            self.molList.append(mol)
            self.Y.append([float(item.expt)])
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):
        return {'smiles':self.smilesList[index],
                'mol':self.molList[index],
                'y':self.Y[index]}

# AGBT: Table 1, Lipophilicity
# (following MoleculeNet, set different random seeds and follow the same procedure ten times to obtain ten different data splitting)
class Lipophilicity(Dataset):
    def __init__(self, path):
        self.path = path
        data = pd.read_csv(os.path.join(path, 'Lipophilicity.csv'))
        self.smilesList,self.molList,self.Y = [],[],[]
        for item in data.itertuples():
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(item.smiles), canonical=True)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f'error in loading molecule with id {item.iupac}')
                continue
            self.smilesList.append(smi)
            self.molList.append(mol)
            self.Y.append([float(item.exp)])
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):
        return {'smiles':self.smilesList[index],
                'mol':self.molList[index],
                'y':self.Y[index]}


# ImageMol: 10 GPCR-related activity regression datasets
class MPP(Dataset):
    def __init__(self, path):
        tmp = pd.read_csv(path)
        self.smilesList = [Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True) for smi in tmp.smiles.tolist()]
        self.molList = [Chem.MolFromSmiles(i) for i in self.smilesList]
        self.Y = [[float(j) for j in i.split()] if isinstance(i,str) else [float(i)] for i in tmp.label.tolist()]
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):
        return {'smiles':self.smilesList[index],
                'mol':self.molList[index],
                'y':self.Y[index]}

class MoleculeNetDC(Dataset):
    def __init__(self, name, type):
        assert type in ['train','valid','test']
        if name=='QM7':
            _,tmp,_ = deepchem.molnet.load_qm7(splitter='scaffold', transformers=[])
        elif name=='QM8':
            _,tmp,_ = deepchem.molnet.load_qm8(splitter='scaffold', transformers=[])
        elif name=='QM9':
            _,tmp,_ = deepchem.molnet.load_qm9(splitter='scaffold', transformers=[])
        elif name=='ESOL':
            _,tmp,_ = deepchem.molnet.load_delaney(splitter='scaffold', transformers=[])
        elif name=='FreeSolv':
            _,tmp,_ = deepchem.molnet.load_sampl(splitter='scaffold', transformers=[])
        elif name=='Lipophilicity':
            _,tmp,_ = deepchem.molnet.load_lipo(splitter='scaffold', transformers=[])
        elif name=='PCBA':
            _,tmp,_ = deepchem.molnet.load_pcba(splitter='scaffold')
        elif name=='MUV':
            _,tmp,_ = deepchem.molnet.load_muv(splitter='scaffold')
        elif name=='HIV':
            _,tmp,_ = deepchem.molnet.load_hiv(splitter='scaffold')
        elif name=='BACE':
            _,tmp,_ = deepchem.molnet.load_bace_classification(splitter='scaffold')
        elif name=='BBBP':
            _,tmp,_ = deepchem.molnet.load_bbbp(splitter='scaffold')
        elif name=='Tox21':
            _,tmp,_ = deepchem.molnet.load_tox21(splitter='scaffold')
        elif name=='ToxCast':
            _,tmp,_ = deepchem.molnet.load_toxcast(splitter='scaffold')
        elif name=='SIDER':
            _,tmp,_ = deepchem.molnet.load_sider(splitter='scaffold')
        elif name=='ClinTox':
            _,tmp,_ = deepchem.molnet.load_clintox(splitter='scaffold')

        if type=='train':
            data = tmp[0]
        elif type=='valid':
            data = tmp[1]
        else:
            data = tmp[2]

        self.smilesList = [Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True) for smi in data.ids]
        self.molList = [Chem.MolFromSmiles(i) for i in self.smilesList]
        self.Y = data.y.tolist()
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):
        return {'smiles':self.smilesList[index],
                'mol':self.molList[index],
                'y':self.Y[index]}

import lmdb
class UniMolMoleculeNetDS(Dataset):
    def __init__(self, name, type):
        assert type in ['train','valid','test']
        if name=='QM7':
            path = './datasets/UniMol/molecular_property_prediction/qm7dft/'
        elif name=='QM8':
            path = './datasets/UniMol/molecular_property_prediction/qm8dft/'
        elif name=='QM9':
            path = './datasets/UniMol/molecular_property_prediction/qm9dft/'
        elif name=='ESOL':
            path = './datasets/UniMol/molecular_property_prediction/esol/'
        elif name=='FreeSolv':
            path = './datasets/UniMol/molecular_property_prediction/freesolv/'
        elif name=='Lipophilicity':
            path = './datasets/UniMol/molecular_property_prediction/lipo/'
        elif name=='PCBA':
            path = './datasets/UniMol/molecular_property_prediction/pcba/'
        elif name=='MUV':
            path = './datasets/UniMol/molecular_property_prediction/muv/'
        elif name=='HIV':
            path = './datasets/UniMol/molecular_property_prediction/hiv/'
        elif name=='BACE':
            path = './datasets/UniMol/molecular_property_prediction/bace/'
        elif name=='BBBP':
            path = './datasets/UniMol/molecular_property_prediction/bbbp/'
        elif name=='Tox21':
            path = './datasets/UniMol/molecular_property_prediction/tox21/'
        elif name=='ToxCast':
            path = './datasets/UniMol/molecular_property_prediction/toxcast/'
        elif name=='SIDER':
            path = './datasets/UniMol/molecular_property_prediction/sider/'
        elif name=='ClinTox':
            path = './datasets/UniMol/molecular_property_prediction/clintox/'

        env = lmdb.open(
            os.path.join(path, f"{type}.lmdb"),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        txn = env.begin()
        keys = list(txn.cursor().iternext(values=False))
        
        self.smilesList,self.molList,self.Y = [],[],[]
        for idx in keys:
            datapoint_pickled = txn.get(idx)
            data = pickle.loads(datapoint_pickled)
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(data['smi']), canonical=False)
            self.smilesList.append(smi)
            self.molList.append(Chem.MolFromSmiles(smi))
            self.Y.append(list(data['target']))
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):
        return {'smiles':self.smilesList[index],
                'mol':self.molList[index],
                'y':self.Y[index]}

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSeq2SeqLM
from rdkit.Chem import Lipinski,Descriptors

import time,traceback
from func_timeout import func_set_timeout

@func_set_timeout(3)
def cal_mol_structure(mol):
    molAH = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(molAH, randomSeed=int(random.random()*10000), maxAttempts=1)
    statis = Chem.MolToMolBlock(molAH)
    return res,statis


from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.AtomPairs import Pairs
def cal_mol_fingerprint(fpType, mol):
    assert fpType in ['[FPR:MACCS]', '[FPR:Toplogical]', '[FPR:ECFP]', '[FPR:FCFP]', '[FPR:Avalon]']
    if fpType=='[FPR:MACCS]':
        return AllChem.GetMACCSKeysFingerprint(mol).ToBitString() # 167 bits
    elif fpType=='[FPR:Toplogical]':
        return FingerprintMols.FingerprintMol(mol, minPath=1, maxPath=7, fpSize=176, 
                                              bitsPerHash=2, useHs=True, tgtDensity=0.0, minSize=128).ToBitString() # 176 bits
    elif fpType=='[FPR:ECFP]':
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 176).ToBitString() # 176 bits
    elif fpType=='[FPR:FCFP]':
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 176, useChirality=True, useFeatures=True).ToBitString() # 176 bits
    elif fpType=='[FPR:Avalon]':
        return pyAvalonTools.GetAvalonFP(mol, nBits=176).ToBitString() # 176 bits

class HuggingfaceNoisingCollateFunc_final: # for GPT
    def __init__(self, bertDir, seqMaxLen, k=16, 
                 mlm_p=0.15, glm_p=0.1, pcglm_p=0.2):
        self.tokenizer = AutoTokenizer.from_pretrained(bertDir, trust_remote_code=True, do_lower_case=False)
        self.tknList = list(self.tokenizer.get_vocab().keys())
        self.seqMaxLen = seqMaxLen
        self.train = False
        with open('./RDKit_LipinskiAttrs.pkl', 'rb') as f:
            self.LipinskiAttrs = pickle.load(f)
        with open('./RDKit_DescriptorsAttrs.pkl', 'rb') as f:
            self.DescriptorsAttrs = pickle.load(f)
        self.k = k
        self.mlm_p = mlm_p

        self.noiseObj = ['smi','pro','val']
        self.noiseStg = ['sMLM','sPLM','sPPLM','sGLM','sPGLM', 'sPPM',
                         'mMLM','mPLM','mPPLM',       'mPGLM', 'mPPM']
        self.destroyStg = []
        self.restorePass = False

    def __call__(self, data):
        if self.restorePass:
            return None
        # drop the items which are too long
        data = [i for i in data if (i['mol'] is not None) and (len(i['smiles'])<384) and ('.' not in i['smiles'])]
        a,b = random.choices(range(len(data)+1), k=2)
        if a>b: a,b = b,a
        data_pc = data[:a] # for physicochemical property prediction
        data_st = data[a:b] # for structure property prediction
        data_fp = data[b:] # for fingerprint property prediction

        # calculate the structure properties
        resList,stPro,stVal = self.get_calculated_structure_by_rdkit(data_st)
        isUsed = [i>-1 and len(j)>0 for i,j in zip(resList,stPro)]
        data_pc += [j for i,j in zip(isUsed,data_st) if not i]
        data_st = [j for i,j in zip(isUsed,data_st) if i]
        stPro,stVal = [j for i,j in zip(isUsed,stPro) if i],[j for i,j in zip(isUsed,stVal) if i]

        # calculate the fingerprint properties
        resList,fpPro,fpVal = self.get_calculated_fingerprint_by_rdkit(data_fp, PAIR=True)
        data_pc += [j for i,j in zip(resList,data_fp) if i<0]
        data_fp = [j for i,j in zip(resList,data_fp) if i>0]
        fpPro,fpVal = [j for i,j in zip(resList,fpPro) if i>0],[j for i,j in zip(resList,fpVal) if i>0]

        # calculate the physicochemical properties
        resList,pcPro,pcVal = self.get_calculated_properties_by_rdkit(data_pc, self.k, PAIR=True)
        data_pc = [j for i,j in zip(resList,data_pc) if i>0]
        pcPro,pcVal = [j for i,j in zip(resList,pcPro) if i>0],[j for i,j in zip(resList,pcVal) if i>0]

        # merge the physicochemical, structure, fingerprint properties
        smiles = [['[SUB]' if j=='.' else j for j in i['smiles']] for i in data_pc+data_st+data_fp]
        properties = pcPro + stPro + fpPro
        values = pcVal + stVal + fpVal

        strategies, source_smiles,source_properties,source_values, target_smiles,target_properties,target_values = self.prepare_nosed_seq2seq_batch(smiles, properties, values, 0.1 if self.train else 0.0)

        if self.train:
            padding = 'longest'
        else:
            padding = 'max_length'
        
        source,target = [],[]
        sequence,label = [],[]
        for idx in range(len(strategies)):
            s,i,j,k = strategies[idx],source_smiles[idx],source_properties[idx],source_values[idx]
            source.append( " ".join([s]+i+["[SEP]"]+sum([[c]+list(v)+[';'] for c,v in zip(j,k)],[])+['[SOS]']).replace('[ M A S K ]','[MASK]').replace('[ V A L U E ]','[VALUE]').replace('[ X Y Z ]','[XYZ]').replace('[ B I T S ]', '[BITS]').replace('[ S U B ]', '[SUB]') )

            i,j,k = target_smiles[idx],target_properties[idx],target_values[idx]
            if 'MLM' in s:
                tmp = i+['[SEP]']
                for idx_,(jj,kk) in enumerate(zip(source_properties[idx],source_values[idx])):
                    if jj=='[MASK]':
                        tmp += [j[0]]
                        j = j[1:]
                    if ('[MASK]' in kk) or (kk=='[VALUE]'):
                        while len(k[0])==0: k=k[1:]
                        v = k[0]
                        tmp += list(v)
                        if len(v)>1: tmp += [';']
                        k = k[1:]
                target.append( " ".join(tmp+['[EOS]']).replace('[ S U B ]', '[SUB]') )
            elif ('PPM' in s) or ('mPGLM_smi_val' in s) or ('SPM' in s): # only values in target
                target.append( " ".join(i+['[SEP]']+sum([list(v)+[';'] for v in k],[])+['[EOS]']).replace('[ S U B ]', '[SUB]') )
            elif 'mPGLM_smi_pro' in s: # only properties in target
                target.append( " ".join(i+['[SEP]']+j+['[EOS]']).replace('[ S U B ]', '[SUB]') )
            else:
                target.append( " ".join(i+["[SEP]"]+sum([[c]+list(v)+[';'] for c,v in zip(j,k)],[])+['[EOS]']).replace('[ S U B ]', '[SUB]') )

            # using for decoder-only model
            sequence.append( source[-1] + " " + target[-1] )

            if len(self.destroyStg)>0:
                for stg in self.destroyStg:
                    if stg in s:
                        target[-1] = " ".join([random.choice(self.tknList) for i in target[-1].split()])

            label.append( " ".join(["[PAD]" for i in source[-1].split(' ')]) + " " + target[-1] )

        if len(sequence)==0:
            print('fk empty!!!')
            sequence = [""]
            label = [""]

        batch = self.tokenizer(sequence, return_tensors='pt', max_length=self.seqMaxLen, padding=padding, truncation=True)
        label = self.tokenizer(label, return_tensors='pt', max_length=self.seqMaxLen, padding=padding, truncation=True)
        batch['labels'] = label['input_ids']

        if 'token_type_ids' in batch:
            batch.pop('token_type_ids')

        return {'smiles':smiles, 'conditions':properties, 'properties':values, 'strategies':strategies, 
                'batch':batch, 'source':source, 'target':target}
    def get_calculated_properties_by_rdkit(self, data, k, PAIR=False):
        resList,proList,valList = [],[],[]
        mol_ = None
        for idx,i in enumerate(data):
            mol = i['mol']
            if PAIR and random.random()>0.5:
                mol_ = random.choice(data)['mol']
            
            try:
                pro,val = [],[]
                k1,k2 = min(np.random.randint(0,k//2+1),len(self.LipinskiAttrs)),min(np.random.randint(0,k//2+1),len(self.DescriptorsAttrs))
                if k1+k2==0:
                    tmp = [0,1]
                    random.shuffle(tmp)
                    k1 += tmp[0]
                    k2 += tmp[1]

                tmp = random.sample(self.LipinskiAttrs, k1)
                pro += [f"[RDKit:{k}]" for k in tmp]
                if mol_ is None:
                    val += [f"{getattr(Lipinski, k)(mol):.3f}" for k in tmp]
                else:
                    val += [f"{getattr(Lipinski, k)(mol)-getattr(Lipinski, k)(mol_):.3f}" for k in tmp]
                tmp = random.sample(self.DescriptorsAttrs, k2)

                pro += [f"[RDKit:{k}]" for k in tmp]
                if mol_ is None:
                    val += [f"{getattr(Descriptors, k)(mol):.3f}" for k in tmp]
                else:
                    val += [f"{getattr(Descriptors, k)(mol)-getattr(Descriptors, k)(mol_):.3f}" for k in tmp]
                    data[idx]['smiles'] = data[idx]['smiles']+'.'+Chem.MolToSmiles(mol_, doRandom=True, canonical=False, isomericSmiles=True)
                res = 1
            except:
                pro,val = [],[]
                res = -1

            resList.append(res)
            proList.append(pro)
            valList.append(val)
            
            mol_ = None
            
        return resList,proList,valList
    def get_calculated_structure_by_rdkit(self, data, eps=0.001):
        resList,proList,valList = [],[],[]
        for i in data:
            try:
                res,statis = cal_mol_structure(i['mol'])
                res = 1
            except:
                res,statis = -1,None
                traceback.print_exc()
                print(f'FUCKING while obtaining the structure of ', end='')
                if 'cid' in i:
                    print(i['cid'], end=' ')
                else:
                    print(i['gid'], end=' ')
                print('with SMILES= ', i['smiles'], '!!!')
                print()
                gc.collect()
            resList.append(res)

            if statis is not None:
                atom3dInfo = re.findall(r'(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s(\w+)\s+', statis)
                xyzArr = np.array([[float(j) for j in i[:3]] for i in atom3dInfo if i[-1]!='H'], dtype=np.float32)
                
                # [d1,d2,d3,loc] or [dihA1,dihA2,d3,loc]
                stType = random.choice(['DDD','AAD','XYZ'])
                if len(xyzArr)>0:
                    xyzArr += np.random.random(xyzArr.shape)*eps
                    xyzArr = xyzEncode(xyzArr, stType)

                if stType=='XYZ':
                    pro = [f"[(XYZ)ATOM:{i[-1]}]" for i in atom3dInfo if i[-1]!='H']
                if stType=='DDD':
                    pro = [f"[(DDD)ATOM:{i[-1]}]" for i in atom3dInfo if i[-1]!='H']
                elif stType=='AAD':
                    pro = [f"[(AAD)ATOM:{i[-1]}]" for i in atom3dInfo if i[-1]!='H']
                val = [",".join([f"{j:.3f}" for j in i[:3]]) for i in xyzArr]
            else:
                pro,val = [],[]
            proList.append(pro)
            valList.append(val)

        return resList,proList,valList
    def get_calculated_fingerprint_by_rdkit(self, data, PAIR=False):
        resList,proList,valList = [],[],[]
        mol_ = None
        for idx,i in enumerate(data):
            mol = i['mol']
            if PAIR and random.random()>0.5:
                mol_ = random.choice(data)['mol']

            try:
                fpType = random.choice(['[FPR:MACCS]', '[FPR:Toplogical]', '[FPR:ECFP]', '[FPR:FCFP]', '[FPR:Avalon]'])
                if mol_ is None:
                    bits = cal_mol_fingerprint(fpType, mol)
                else:
                    bits = "".join(["0" if b==b_ else "1" for b,b_ in zip(cal_mol_fingerprint(fpType, mol),cal_mol_fingerprint(fpType, mol_))])
                    data[idx]['smiles'] = data[idx]['smiles']+'.'+Chem.MolToSmiles(mol_, doRandom=True, canonical=False, isomericSmiles=True)
                res = 1
            except:
                fpType = ""
                bits = ""
                res = -1
                gc.collect()

            resList.append(res)
            proList.append([fpType])
            valList.append([bits])
            
            mol_ = None
            
        return resList,proList,valList

    def prepare_nosed_seq2seq_batch(self, smiles, properties, values, pro_r=0.1):
        strategies = []
        source_smiles,source_properties,source_values = [],[],[]
        target_smiles,target_properties,target_values = [],[],[]

        for idx in range(len(smiles)):
            smi,pro,val = smiles[idx],properties[idx],values[idx]
            
            isStruPre,isFprPre = False,False
            if 'ATOM' in pro[0]:
                valueMASK = '[XYZ]'
                isStruPre = True
            elif 'FPR' in pro[0]:
                valueMASK = '[BITS]'
                isFprPre = True
            else:
                valueMASK = '[VALUE]'
            
            # random replace CONDITION with CUSPRO
            pro = [i if random.random()>pro_r else '[CUSPRO]' for i in pro]
            ssmi,spro,sval = [],[],[]
            tsmi,tpro,tval = [],[],[]

            stg = random.choice(self.noiseStg)
            # random mask: random mask some tokens
            if stg=='sMLM':
                obj = random.choice(self.noiseObj)
                if obj=='smi':
                    strategies.append('[sMLM_smi]')
                    mask = [False if random.random()>self.mlm_p else True for i in smi]
                    ssmi = [i if not m else '[MASK]' for m,i in zip(mask,smi)]
                    tsmi = [i for m,i in zip(mask,smi) if m]

                    spro = pro
                    sval = val
                elif obj=='pro':
                    strategies.append('[sMLM_pro]')
                    mask = [False if random.random()>self.mlm_p else True for i in pro]
                    spro = [i if not m else '[MASK]' for m,i in zip(mask,pro)]
                    tpro = [i for m,i in zip(mask,pro) if m]

                    ssmi = smi
                    sval = val
                else:
                    strategies.append('[sMLM_val]')
                    if random.random()<0.5:
                        mask = [False if random.random()>self.mlm_p else True for i in val]
                        sval = [i if not m else valueMASK for m,i in zip(mask,val)]
                        tval = [i for m,i in zip(mask,val) if m]
                    else:
                        mask = [[False if random.random()>self.mlm_p else True for j in i] for i in val]
                        sval = ["".join([ii if not mm else '[MASK]' for mm,ii in zip(m,i)]) for m,i in zip(mask,val)]
                        tval = ["".join([ii for mm,ii in zip(m,i) if mm]) for m,i in zip(mask,val)]

                    ssmi = smi
                    spro = pro
            elif stg=='mMLM':
                strategies.append('[mMLM]')
                mask = [False if random.random()>self.mlm_p else True for i in smi]
                ssmi = [i if not m else '[MASK]' for m,i in zip(mask,smi)]
                tsmi = [i for m,i in zip(mask,smi) if m]

                mask = [False if random.random()>self.mlm_p else True for i in pro]
                spro = [i if not m else '[MASK]' for m,i in zip(mask,pro)]
                tpro = [i for m,i in zip(mask,pro) if m]

                if random.random()<0.5:
                    mask = [False if random.random()>self.mlm_p else True for i in val]
                    sval = [i if not m else valueMASK for m,i in zip(mask,val)]
                    tval = [i for m,i in zip(mask,val) if m]
                else:
                    mask = [[False if random.random()>self.mlm_p else True for j in i] for i in val]
                    sval = ["".join([ii if not mm else '[MASK]' for mm,ii in zip(m,i)]) for m,i in zip(mask,val)]
                    tval = ["".join([ii for mm,ii in zip(m,i) if mm]) for m,i in zip(mask,val)]
            
            # permulation noise: random shuffle some tokens
            elif stg=='sPLM':
                obj = random.choice(self.noiseObj)
                if obj=='smi':
                    strategies.append('[sPLM_smi]')
                    tsmi = [i for i in smi]
                    random.shuffle(smi)
                    ssmi = [i for i in smi]

                    spro = pro
                    sval = val
                else:
                    strategies.append('[sPLM_pro_val]')
                    tpro,tval = [i for i in pro],[i for i in val]
                    if isFprPre:
                        tmp = list(val[0])
                        random.shuffle(tmp)
                        tmp = ''.join(tmp)
                        sval = [tmp]
                    else:
                        random.shuffle(val)
                        sval = [i for i in val]

                    ssmi = smi
                    spro = pro
            elif stg=='mPLM':
                strategies.append('[mPLM]')
                tsmi = [i for i in smi]
                random.shuffle(smi)
                ssmi = [i for i in smi]

                tpro = [i for i in pro]
                random.shuffle(pro)
                spro = [i for i in pro]

                tval = [i for i in val]
                if isFprPre:
                    tmp = list(val[0])
                    random.shuffle(tmp)
                    tmp = ''.join(tmp)
                    sval = [tmp]
                else:
                    random.shuffle(val)
                    sval = [i for i in val]

            elif stg=='sPPLM':
                obj = random.choice(self.noiseObj)
                if obj=='smi':
                    strategies.append('[sPPLM_smi]')
                    s,e = random.sample(range(len(smi)+1), 2)
                    if s>e: s,e = e,s
                    part = smi[s:e]
                    tsmi = [i for i in part]
                    random.shuffle(part)
                    ssmi = smi[:s] + [i for i in part] + smi[e:]

                    spro = pro
                    sval = val
                else:
                    strategies.append('[sPPLM_pro_val]')
                    if isFprPre:
                        s,e = random.sample(range(len(val[0])+1), 2)
                        if s>e: s,e = e,s
                        part = list(zip(pro, [val[0][s:e]]))
                        tpro,tval = [i[0] for i in part],[i[1] for i in part]
                        tmp = list(part[0][1])
                        random.shuffle(tmp)
                        tmp = ''.join(tmp)
                        sval = [val[0][:s] + tmp + val[0][e:]]
                    else:
                        s,e = random.sample(range(len(val)+1), 2)
                        if s>e: s,e = e,s
                        part = list(zip(pro[s:e],val[s:e]))
                        tpro,tval = [i[0] for i in part],[i[1] for i in part]
                        random.shuffle(part)
                        sval = val[:s] + [i[1] for i in part] + val[e:]

                    ssmi = smi
                    spro = pro
            elif stg=='mPPLM':
                strategies.append('[mPPLM]')
                s,e = random.sample(range(len(smi)+1), 2)
                if s>e: s,e = e,s
                part = smi[s:e]
                tsmi = [i for i in part]
                random.shuffle(part)
                ssmi = smi[:s] + [i for i in part] + smi[e:]

                if isFprPre:
                    s,e = random.sample(range(len(val[0])+1), 2)
                    if s>e: s,e = e,s
                    part = list(zip(pro, [val[0][s:e]]))
                    tpro,tval = [i[0] for i in part],[i[1] for i in part]
                    tmp = list(part[0][1])
                    random.shuffle(tmp)
                    tmp = ''.join(tmp)
                    sval = [val[0][:s] + tmp + val[0][e:]]
                else:
                    s,e = random.sample(range(len(val)+1), 2)
                    if s>e: s,e = e,s
                    part = list(zip(pro[s:e],val[s:e]))
                    tpro,tval = [i[0] for i in part],[i[1] for i in part]
                    random.shuffle(part)
                    sval = val[:s] + [i[1] for i in part] + val[e:]

                spro = pro

            # conditional generation noise: generate the next tokens based on the previous tokens
            elif stg=='sGLM':
                obj = 'smi'
                if obj=='smi':
                    strategies.append('[sGLM_smi]')
                    tsmi = smi
                    ssmi.append('[SPAN]')

                    spro = pro
                    sval = val
                
            elif stg=='sPGLM':
                obj = 'smi'
                if obj=='smi':
                    strategies.append('[sPGLM_smi]')
                    # i = random.choice(range(len(smi)))
                    # ssmi = smi[:i] + ['[SPAN]']
                    # tsmi = smi[i:]

                    s,e = random.sample(range(len(smi)+1), 2)
                    if s>e: s,e = e,s
                    ssmi = smi[:s] + ['[SPAN]'] + smi[e:]
                    tsmi = smi[s:e]

                    spro = pro
                    sval = val

            elif stg=='mPGLM':
                i = random.choice(range(len(smi)))
                ssmi = smi[:i] + ['[SPAN]']
                tsmi = smi[i:]

                if random.random()<0.5:
                    strategies.append('[mPGLM_smi_pro]')
                    i = random.choice(range(len(pro)))
                    spro = pro[:i] + ['[MASK]' for j in pro[i:]]
                    tpro = pro[i:]

                    sval = val
                else:
                    strategies.append('[mPGLM_smi_val]')
                    i = random.choice(range(len(val)))
                    sval = val[:i] + [valueMASK for j in val[i:]]
                    tval = val[i:]

                    spro = pro
            else:
                # properties prediction model
                if isStruPre:
                    if 'DDD' in pro[0]:
                        strategies.append('[SPM_DDD]')
                    elif 'AAD'in pro[0]:
                        strategies.append('[SPM_AAD]')
                    elif 'XYZ' in pro[0]:
                        strategies.append('[SPM_XYZ]')
                    ssmi = smi
                    spro = pro
                    sval = [valueMASK for i in val]
                    tval = val
                elif stg=='sPPM':
                    strategies.append('[sPPM]')
                    ssmi = smi
                    i = random.choice(range(len(pro)))
                    spro = [pro[i]]
                    sval = [valueMASK]
                    tval = [val[i]]
                elif stg=='mPPM':
                    strategies.append('[mPPM]')
                    ssmi = smi
                    ii = random.sample(range(len(pro)), random.choice(range(1,len(pro)+1)))
                    spro = [pro[i] for i in ii]
                    sval = [valueMASK for i in ii]
                    tval = [val[i] for i in ii]
                
            source_smiles.append(ssmi)
            source_properties.append(spro)
            source_values.append(sval)
            target_smiles.append(tsmi)
            target_properties.append(tpro)
            target_values.append(tval)

            if '[SUB]' in smi:
                strategies[-1] = strategies[-1][:-1]+'(PAIR)'+strategies[-1][-1:]

        return strategies, source_smiles,source_properties,source_values,target_smiles,target_properties,target_values

class InferenceCollateFunc:
    def __init__(self, bertDir, seqMaxLen, prompt):
        self.tokenizer = AutoTokenizer.from_pretrained(bertDir, trust_remote_code=True, do_lower_case=False)
        self.seqMaxLen = seqMaxLen
        self.prompt = prompt
        self.padding = 'max_length'
    def __call__(self, data):
        source = [self.prompt.replace('[SMILES]', " ".join(list(i['smiles']))) for i in data]
        batch = self.tokenizer(source, return_tensors='pt', max_length=self.seqMaxLen, padding=self.padding, truncation=True)
        if 'token_type_ids' in batch:
            batch.pop('token_type_ids')

        return {'batch':batch}

class FinetuneCollateFunc_final:
    def __init__(self, bertDir, seqMaxLen, prompt, normVec=None, randomSMILES=True, 
                 useStructure=False, seq2seq=False, use_all=False, use_graph=False):
        self.tokenizer = AutoTokenizer.from_pretrained(bertDir, trust_remote_code=True, do_lower_case=False)
        self.tokenizer.padding_side = 'left'
        self.cusProNum = prompt.count('CUSPRO:')
        self.tokenizer.add_tokens([f"[CUSPRO:{i}]" for i in range(self.cusProNum)])
        self.seqMaxLen = seqMaxLen
        self.prompt = prompt
        self.randomSMILES = randomSMILES
        self.useStructure = useStructure
        self.train = False
        self.seq2seq = seq2seq
        self.use_all = use_all
        self.use_graph = use_graph
        self.normVec = normVec
    def __call__(self, data):
        if self.train and self.randomSMILES:
            smilesList = [Chem.MolToSmiles(i['mol'], doRandom=True, canonical=False, isomericSmiles=True) for i in data]
            if self.useStructure: molList = [Chem.MolFromSmiles(i) for i in smilesList]
        else:
            smilesList = [i['smiles'] for i in data]
            if self.useStructure: molList = [i['mol'] for i in data]

        source = [self.prompt.replace('[SMILES]', " ".join(list(smi))) for smi in smilesList]
        
        if self.useStructure:
            source = [s.replace('[(DDD)STRUCTURE]'," ".join([f"[(DDD)ATOM:{a.GetSymbol()}]" for a in mol.GetAtoms() if a.GetSymbol()!='H'])) \
                       .replace('[(AAD)STRUCTURE]'," ".join([f"[(AAD)ATOM:{a.GetSymbol()}]" for a in mol.GetAtoms() if a.GetSymbol()!='H'])) \
                       .replace('[(XYZ)STRUCTURE]'," ".join([f"[(XYZ)ATOM:{a.GetSymbol()}]" for a in mol.GetAtoms() if a.GetSymbol()!='H'])) for s,mol in zip(source,molList)]

        if isinstance(data[0]['y'], list):
            target = ["[SEP] " + " ".join(sum([list(f"{v:.3f}")+[';'] for v in i['y']],[])) + " [EOS]" for i in data]
        else:
            target = [" ".join(['[SEP]']+list(f"{i['y']:.3f}")+[';','[EOS]']) for i in data]

        if self.train:
            padding = 'longest'
        else:
            padding = 'max_length'

        if self.seq2seq:
            sequence = [s+" "+t for s,t in zip(source,target)]
            label = [" ".join(["[PAD]" for i in s.split(' ')]) + " " + t for s,t in zip(source,target)]
        else:
            sequence = source
            label = target

        batch = self.tokenizer(sequence, return_tensors='pt', max_length=self.seqMaxLen, padding=padding, truncation=True)
        if self.seq2seq:
            label = self.tokenizer(label, return_tensors='pt', max_length=self.seqMaxLen, padding=padding, truncation=True)
            batch['labels'] = label['input_ids']
        
        if 'token_type_ids' in batch:
            batch.pop('token_type_ids')

        y = np.array([i['y'] for i in data], dtype=np.float32)
        if self.normVec is not None:
            y = (y-self.normVec[0]) / self.normVec[1]
        
        res = {'batch':batch, 'y':torch.tensor(y, dtype=torch.float32)}

        if self.use_all:
            sequence = [s+" "+t for s,t in zip(source,target)]
            label = [" ".join(["[PAD]" for i in s.split(' ')]) + " " + t for s,t in zip(source,target)]
            batch_seq2seq = self.tokenizer(sequence, return_tensors='pt', max_length=self.seqMaxLen, padding=padding, truncation=True)
            label = self.tokenizer(label, return_tensors='pt', max_length=self.seqMaxLen, padding=padding, truncation=True)
            batch_seq2seq['labels'] = label['input_ids']
            if 'token_type_ids' in batch_seq2seq:
                batch_seq2seq.pop('token_type_ids')

            sequence = source
            label = target
            batch_seq2val = self.tokenizer(sequence, return_tensors='pt', max_length=self.seqMaxLen, padding=padding, truncation=True)
            label = self.tokenizer(label, return_tensors='pt', max_length=self.seqMaxLen, padding=padding, truncation=True)
            batch_seq2val['labels'] = label['input_ids']

            if 'token_type_ids' in batch_seq2val:
                batch_seq2val.pop('token_type_ids')

            res['batch_seq2seq'] = batch_seq2seq
            res['batch_seq2val'] = batch_seq2val
        
        if self.use_graph:
            maxAtomNum = np.max([len(mol.GetAtoms()) for mol in molList])
            sequence = [" ".join([f"[(AAD)ATOM:{a.GetSymbol()}]" for a in mol.GetAtoms() if a.GetSymbol()!='H']) for mol in molList]
            isHs = [[a.GetSymbol()=='H' for a in mol.GetAtoms()] for mol in molList]
            res['batch_atom'] = self.tokenizer(sequence, return_tensors='pt', max_length=maxAtomNum, padding='max_length', truncation=True)

            # get adjacency matrix
            adjArr = np.zeros((len(molList),maxAtomNum,maxAtomNum), dtype=np.float32)
            graph_attention_mask = batch['attention_mask'][...,None].bool() & batch['attention_mask'][:,None].bool() # B,L,L
            L = graph_attention_mask.shape[1]
            for idx,(smi,mol,isH) in enumerate(zip(smilesList,molList,isHs)):
                isH = np.array(isH, dtype=bool)
                adj = Chem.rdmolops.GetAdjacencyMatrix(mol)[~isH][:,~isH]
                adj = adj + np.eye(len(adj))
                ma = min(adj.shape[0],maxAtomNum)
                if adj.shape[0]==0:
                    continue
                adjArr[idx][-ma:,-ma:] = adj[:ma,:ma]
                
                preL,an = min(len(smi)+6,L),adj.shape[1]
                ma = min(adj.shape[0],L-preL)
                graph_attention_mask[idx][preL:preL+ma,preL:preL+ma] = torch.from_numpy(adj[:ma,:ma].astype(bool))


            res['adjArr'] = torch.tensor(adjArr, dtype=torch.float32)
            res['graph_attention_mask'] = graph_attention_mask
            
        return res 
