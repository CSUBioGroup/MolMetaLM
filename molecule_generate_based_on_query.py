from rdkit import Chem
from rdkit.Chem import AllChem
import os,traceback
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--pretrainPath', default='', type=str)
parser.add_argument('--method', type=str, default='structure', choices=['structure','fingerprint'])
parser.add_argument('--topK', type=int, default=10)
args = parser.parse_args()

from utils import *
from DL_ClassifierModel import *
from matplotlib import pyplot as plt
import pubchempy as pcp
import seaborn as sns
import rdkit

if __name__=='__main__':
    collater = HuggingfaceNoisingCollateFunc_final(bertDir='./bertModel/Tokenizer_final', seqMaxLen=512, k=64)
    tkn2id = collater.tokenizer.get_vocab()
    config = AutoConfig.from_pretrained("./bertModel/cus-llama2-base", trust_remote_code=True, use_flash_attention_2=True)
    
    backbone = MolMetaLM(config, tkn2id, maxGenLen=512).cuda()
    model = HuggingfaceSeq2SeqLanguageModel(backbone, collateFunc=collater, AMP=True)
    backbone.alwaysTrain = False
    
    if len(args.pretrainPath)>0:
        model.load(args.pretrainPath)
    else:
        model.model.backbone = AutoModelForCausalLM.from_pretrained('wudejian789/MolMetaLM-base').cuda()
    model.to_eval_mode()

    if args.input.endswith('.sdf') or args.input.endswith('.mol'):
        mol = Chem.MolFromMolFile(args.input)
    else:
        mol = Chem.MolFromSmiles(args.input)
    ref_smi = Chem.MolToSmiles(mol, doRandom=False, canonical=True)
    fpr_ref = AllChem.GetMACCSKeysFingerprint(mol)
    hasConf = len(mol.GetConformers())>0

    print('start generating...')
    molList,simList = [],[]
    for loc in tqdm(range(int(args.topK))):
        while True:
            if args.method=='structure':
                eps = 0.001

                smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False, isomericSmiles=True)
                smi2molOrder = [int(j) for j in mol.GetProp('_smilesAtomOutputOrder')[1:-2].split(',')]
                if not hasConf:
                    mol.RemoveAllConformers()
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol)
                    AllChem.MMFFOptimizeMolecule(mol)
                    mol = AllChem.RemoveHs(mol)
                
                atomArr = np.array([i.GetSymbol() for i in mol.GetAtoms()])[smi2molOrder].tolist()
                xyzArr = mol.GetConformer().GetPositions()[smi2molOrder]

                stType = random.choice(['DDD','AAD','XYZ'])
                xyzArr += np.random.random(xyzArr.shape)*eps

                pro = [f"[({stType})ATOM:{a}]" for a in atomArr]
                val = [",".join([f"{j:.3f}" for j in p]) for p in xyzEncode(xyzArr, stType)]

                source = ["[mPGLM_smi_pro] [SPAN] [SEP] " + " ".join([f"[MASK] {' '.join(list(i[1]))} ;" for i in list(zip(pro, val))]) + " [SOS]"]
            elif args.method=='fingerprint':
                resList, proList, valList = collater.get_calculated_fingerprint_by_rdkit([{'mol':mol}])
                source = ["[sGLM_smi] [SPAN] [SEP] " + " ".join([f"{i[0]} {' '.join(list(i[1]))} ;" for i in list(zip(proList[0], valList[0]))]) + " [SOS]"]
            batch = collater.tokenizer(source, return_tensors='pt', max_length=1024, padding='longest', truncation=True)
            if 'token_type_ids' in batch:
                batch.pop('token_type_ids')
            batch = {'batch':batch}
            batch = dict_to_device(batch, 'cuda')

            data = batch['batch']
            target_pre = collater.tokenizer.batch_decode(
                            model.model.backbone.generate(**({k:data[k] for k in data if k!='labels'}), 
                                                          max_length=1024, num_beams=4, length_penalty=model.model.length_penalty, 
                                                          no_repeat_ngram_size=9, do_sample=True), 
                         skip_special_tokens=False)[0]

            gen_smi = target_pre[target_pre.find('[SOS]'):].replace('[SOS]','')
            gen_smi = gen_smi[:gen_smi.find('[SEP]')].replace(' ','')
            gen_mol = Chem.MolFromSmiles(gen_smi)
            
            if gen_mol is not None and (Chem.MolToSmiles(gen_mol, doRandom=False, canonical=True)!=ref_smi):
                break

        if len(gen_mol.GetAtoms())==len(mol.GetAtoms()):
            conf = Chem.Conformer(gen_mol.GetNumAtoms())
            for i in range(len(xyzArr)):
                conf.SetAtomPosition(i, (xyzArr[i]).tolist())
            gen_mol.AddConformer(conf)
            gen_mol = Chem.AddHs(gen_mol, addCoords=True)
        else:
            gen_mol = Chem.AddHs(gen_mol)
            AllChem.EmbedMolecule(gen_mol)
        
        AllChem.MMFFOptimizeMolecule(gen_mol)
        gen_mol = AllChem.RemoveHs(gen_mol)

        fpr_gen = AllChem.GetMACCSKeysFingerprint(gen_mol)
        sim = rdkit.DataStructs.TanimotoSimilarity(fpr_ref,fpr_gen)

        molList.append(gen_mol)
        simList.append(sim)
        
    sortedIdx = np.argsort(simList)[::-1]
    molList,simList = [molList[i] for i in sortedIdx],[simList[i] for i in sortedIdx]

    print("==============================RESULTS==============================")
    for i,(mol,sim) in enumerate(zip(molList,simList)):
        i = i+1
        smi = Chem.MolToSmiles(mol, doRandom=False, canonical=True)
        if len(smi)>55: smi = smi[:50]+'...'
        print(f"{i:3d} {smi:55} {sim:6.5f}")

        with open(f'{args.output}/gen_mol{i}.sdf', 'w') as f:
            f.write(Chem.MolToMolBlock(mol))