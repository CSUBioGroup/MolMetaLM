# MolMetaLM: a Physicochemical Knowledge-Guided Molecular Meta Language Model

## Overview
MolMetaLM is a universal molecular language model based on molecular meta language covering physicochemical knowledge. The molecular meta language is formatted as multiple <S,P,O> knowledge triples sharing the same S to enhance learning the semantic relationships between physicochemical knowledge and molecules. By replacing the P and O with different molecular physicochemical properties, fingerprints, or atom coordinates, the meta language paradigm generates tens of thousands of pretraining tasks. By recovering the token / sequence / order -level noise, MolMetaLM exhibits proficiency in large-scale benchmark evaluations involving property prediction, molecule generation, conformation inference, and molecular optimization.

![image](https://github.com/CSUBioGroup/MolMetaLM/blob/main/figures/framework.png)

## Datasets
### 1. PubChem for pretraining MolMetaLM
A dataset containing 110M molecular SMILES: [https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz)

Please put the extracted file "CID-SMILES" into "datasets/PubChem/CID-SMILES". 

### 2. Benchmark datasets for evaluating MolMetaLM
Download from Google Drive: [https://drive.google.com/file/d/1azyIzIJHw0FvmIPlyLKE27sTpB_AGtqE/view?usp=drive_link](https://drive.google.com/file/d/1azyIzIJHw0FvmIPlyLKE27sTpB_AGtqE/view?usp=drive_link)

Please put the extracted folders AGBT,GPCR,UniMol in "datasets/...".

## Usage
### Requirements
- scikit-learn==1.4.0
- torch==2.3.1
- transformers==4.38.2
- rdkit==2023.9.5
- deepchem==2.8.0

### 1. Train your own MolMetaLM (optional)
#### 1.1 prepare dataset for pretraining
The file only needs to contain the SMILES of molecules, one molecular SMILES per line:
```python
CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C
C1=CC(C(C(=C1)C(=O)O)O)O
CC(CN)O
C1=CC(=C(C=C1[N+](=O)[O-])[N+](=O)[O-])Cl
...
```

or you can also follow us and download the [PubChem](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz) dataset (containing 110M molecular SMILES) for pretraining and 
#### 1.2 pretrain MolMetaLM

```python
python pretrain_MolMetaLM.py --maxSteps 1000 --warmupSteps 100 --dataset ./datasets/smiles.txt --model_size base
```

For PubChem datasets, set param 'dataset' to 'pubchem'; 

For multi nodes/cards training, set param 'ddp' to 'true' and follow the DistributedDataParallel's command (torchrun --nproc_per_node xxx) in Pytorch for ddp training.

### 2. Use MolMetaLM for downstream applicaitons

After training, MolMetaLM can be used to:
#### 2.1 via scripts
a) obtain molecular representation:
```python
python molecule_embedding.py --input_file ./datasets/smiles.txt --output_file ./cache/smiles.pkl
```

b) generate molecules based on query molecule:
```python
# molecular generation based on query molecule
# param method: `structure` for query molecular backbone-based generation; `fingerprint` for query molecular fingerprint-based generation
python molecule_generate_based_on_query.py --input "COc1cccc(/C=N/N2C(=O)c3ccccc3C2=O)c1OC"  --output ./cache \
                                           --method structure --topK 10
```

c) finetune MolMetaLM for MoleculeNet classification tasks:
```python
# param datasets: UniMol_PCBA,UniMol_MUV,UniMol_HIV,UniMol_BACE,UniMol_BBBP,UniMol_Tox21,UniMol_ToxCast,UniMol_SIDER,UniMol_ClinTox
python -u finetune_main_final.py --datasets "UniMol_BBBP" \
                                 --taskType "classification" --modelType "llama2-base"
```

d) finetune MolMetaLM for GPCR-related activity or AGBT's regression tasks:
```python
# param datasets: GPCR_5HT1A,GPCR_5HT2A,GPCR_AA1R,GPCR_AA2AR,GPCR_AA3R,GPCR_CNR2,GPCR_DRD2,GPCR_DRD3,GPCR_HRH3,GPCR_OPRM,
#                 AGBT_LD50,AGBT_IGC50,AGBT_LC50,AGBT_LC50DM,AGBT_LogP,AGBT_FreeSolv,AGBT_Lipophilicity
python -u finetune_main_final.py --datasets "GPCR_5HT1A" \
                                 --taskType "regression" --modelType "llama2-base"
```

Or do more detailed development based on MolMetaLM:
#### 2.2 via codes
Firstly, we need to prepare the pretrained MolMetaLM from huggingface:
```python
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('wudejian789/MolMetaLM-base')
descriptors = [i[7:-1] for i in tokenizer.get_vocab().keys() if i.startswith('[RDKit:')]
model = AutoModelForCausalLM.from_pretrained('wudejian789/MolMetaLM-base').cuda()
model = model.eval()
```
or from your own MolMetaLM:
```python
from utils import *
from DL_ClassifierModel import *

collater = HuggingfaceNoisingCollateFunc_final(bertDir='./bertModel/Tokenizer_final', seqMaxLen=512, k=64)
descriptors = collater.DescriptorsAttrs
tkn2id = collater.tokenizer.get_vocab()

config = AutoConfig.from_pretrained("./bertModel/cus-llama2-base", trust_remote_code=True, use_flash_attention_2=True)

backbone = MolMetaLM(config, tkn2id, maxGenLen=512).cuda()
model = HuggingfaceSeq2SeqLanguageModel(backbone, collateFunc=collater, AMP=True)

model.load('path to your pretrained weight')

tokenizer = collater.tokenizer
model = model.model
```

Then we can further investigate or use MolMetaLM as the same as other huggingface pre-trained models.
We also provide some example codes:

a) obtain molecular representation:
```python
smi = "COc1cc2c(cc1OC)CC([NH3+])C2"
tokenized_smi = tokenizer(" ".join(list(smi)), return_token_type_ids=False, 
                          return_tensors='pt', max_length=512, padding='longest', truncation=True)
emb_smi = model.model(**tokenized_smi).last_hidden_state
print(emb_smi.shape) # batch size, seq length, embedding size
```

b) generate molecules conditioned by multiple properties:
```python
import re
def analysis_property_form_mol(m, s):
    res = []
    for f in [re.findall('\[RDKit:(.*?)\]', i)[0] for i in re.findall('(\[RDKit:[0-9a-zA-Z]*?\][0-9.\s]+?);', s)]:
        res.append( f"[RDKit:{f}] "+" ".join(list(f"{getattr(Descriptors, f)(m):.3f}")))
    return res
source = ["[sGLM_smi] [SPAN] [SEP] [RDKit:NumAromaticRings] 1 0 . 0 0 0 ; [RDKit:MolWt] 6 3 6. 0 0 0 ; [SOS]"]
batch = tokenizer(source, return_tensors='pt', max_length=512, padding='longest', truncation=True, return_token_type_ids=False)
target_pre = tokenizer.batch_decode(model.generate(**batch, max_length=512))[0]
smiles = target_pre[target_pre.find('[SOS]')+6:target_pre.find('[EOS]')-7].replace(' ','')
mol = Chem.MolFromSmiles(smiles)
print("generated smiles:",smiles)
print('properties:', analysis_property_form_mol(mol, source[0]))
```

the "[RDKit:xxx]" in var source can be replaced with other properties (see all of the supported properties by "print(pickle.load(open('RDKit_DescriptorsAttrs.pkl','rb')))").

currently, MolMetaLM supports conditional generation of up to 64 properties simultaneously.

c) optimize molecules based on multiple properties:
```python
from rdkit import DataStructs
ref_smi = "O=C(N[C@@H](C(=O)[O-])c1ccccc1)C1CCC(CNC(=O)[C@@H]2Cc3ccccc3C[NH2+]2)CC1"
ref_mol = Chem.MolFromSmiles(ref_smi)
proList = ["MolLogP","qed","TPSA"] # target property list you want to optimize 
valList = [-1.0,-0.1,10]           # value difference list you want to change
k = 4                              # factor parameter to control the similarity

pros = [f"[RDKit:{i}]" for i in proList]
vals = [" ".join(list(f"{i:.3f}")) for i in valList]
otherCons = [f"[RDKit:{i}]" for i in descriptors if i not in proList]

while True:
    smi_local = Chem.MolToSmiles(ref_mol, doRandom=True)
    source = "[sPGLM_smi(PAIR)] [SMILES] [SUB] [SPAN] [SEP]" + " ".join([f"{p} {v} ;" for p,v in zip(pros,vals)]) + " ".join([f"{p} 0 . 0 0 0 ;" for p in random.sample(otherCons,k)]) + " [SOS]"
    batch = tokenizer([source.replace('[SMILES]', " ".join(list(ref_smi)))], return_token_type_ids=False, 
                      return_tensors='pt', max_length=collater.seqMaxLen, padding='longest', truncation=True)
    batch = dict_to_device(batch, 'cuda')
    target_pre = tokenizer.batch_decode(model.generate(**batch, max_length=1024, 
                                                       do_sample=True, no_repeat_ngram_size=6, num_beams=5, length_penalty=0))[0]
    smi_opted = target_pre[target_pre.find('[SOS]')+6:target_pre.find('[EOS]')-7].replace(' ','')
    mol_opted = Chem.MolFromSmiles(smi_opted)
    if mol_opted is not None:
        break
fp1 = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048, useChirality=False)
fp2 = AllChem.GetMorganFingerprintAsBitVect(mol_opted, 2, nBits=2048, useChirality=False)
sim = DataStructs.TanimotoSimilarity(fp1, fp2)

print(f'original smiles:       ', ref_smi)
print(f'new smiles (sim={sim:.3f}):', smi_opted)

for pro in proList:
    func = getattr(Descriptors, pro)
    print(pro, ':', func(ref_mol)-func(mol_opted))
```

d) predict molecular conformation:
```python
# Note: the length of the structural sequence of large molecules will be too large, beyond the preset sequence length of 512. 
#       therefore, the current model is very unstable when predicting conformation for big molecules (whose num of atoms>20).
stType = "AAD"
smi = "C[N@H+]1[C@@H]2[C@H](O)[C@H]1[C@@H]1N[C@@H]12"
mol = Chem.MolFromSmiles(smi)
print('num of atoms:', len(mol.GetAtoms()))
source = [f"[SPM_{stType}] {' '.join(list(smi))} [SEP] {' '.join(['[('+stType+')ATOM:'+a.GetSymbol()+'] [XYZ] ;' for a in mol.GetAtoms()])} [SOS]"]
batch = tokenizer(source, return_tensors='pt', max_length=512, padding='longest', truncation=True, return_token_type_ids=False)
batch = dict_to_device(dict(**batch), 'cuda')
target_pre = tokenizer.batch_decode(model.generate(**batch, max_length=512, do_sample=False,
                                                   num_beams=1, no_repeat_ngram_size=0))[0]

sIdx,eIdx = target_pre.find('[SOS] [SEP]')+12,target_pre.find('[EOS]')
tmp = [[float(j) for j in i.split(',')] for i in target_pre[sIdx:eIdx].replace(' ','').split(';')[:-1]]
xyzArr_pre = xyzDecode(np.array(tmp), stType=stType, eps=1e-8) # num of atoms, 3
print(xyzArr_pre)
```
