import os,pickle
# os.rename('tokenizer_config.json','tokenizer_config_bak.json')
# os.rename('vocab.txt','vocab_bak.txt')

QuantumChemAttrs = ['[QM9:A_RC]', '[QM9:B_RC]', '[QM9:C_RC]', '[QM9:miu_DM]', '[QM9:alpha_IP]', '[QM9:epsE_HOMO]','[QM9:epsE_LUMO]','[QM9:eps_GAP]','[QM9:R2_ESE]','[QM9:zpve_ZPVE]',
                    '[QM9:Uo_IEat0K]','[QM9:U_IEat298.15K]','[QM9:H_Eat298.15K]','[QM9:G_FEat298.15K]','[QM9:Cv_HCat298.15K]']

# LipinskiAttrs = [f"[RDKit:{i}]" for i in ['FractionCSP3','HeavyAtomCount','NHOHCount','NOCount','NumAliphaticCarbocycles','NumAliphaticHeterocycles','NumAliphaticRings',
#                  'NumAromaticCarbocycles','NumAromaticHeterocycles','NumAromaticRings','NumHAcceptors','NumHDonors','NumHeteroatoms','NumRotatableBonds',
#                  'NumSaturatedCarbocycles','NumSaturatedHeterocycles','NumSaturatedRings','RingCount']]

# DescriptorsAttrs = [f"[RDKit:{i}]" for i in ['ExactMolWt','FpDensityMorgan1','FpDensityMorgan2','FpDensityMorgan3','HeavyAtomMolWt',
#                     'MaxAbsPartialCharge','MaxPartialCharge','MinAbsPartialCharge','MinPartialCharge','MolWt','NumRadicalElectrons','NumValenceElectrons']]

with open('../../RDKit_LipinskiAttrs.pkl', 'rb') as f:
    LipinskiAttrs = [f"[RDKit:{i}]" for i in pickle.load(f)]

with open('../../RDKit_DescriptorsAttrs.pkl', 'rb') as f:
    DescriptorsAttrs = [f"[RDKit:{i}]" for i in pickle.load(f)]

AtomPosition = ['[ATOM:Ac]', '[ATOM:Ag]', '[ATOM:Al]', '[ATOM:Am]', '[ATOM:Ar]', '[ATOM:As]', '[ATOM:At]', '[ATOM:Au]',
                '[ATOM:B]', '[ATOM:Ba]', '[ATOM:Be]', '[ATOM:Bi]', '[ATOM:Bk]', '[ATOM:Br]',
                '[ATOM:C]', '[ATOM:Ca]', '[ATOM:Cd]', '[ATOM:Ce]', '[ATOM:Cf]', '[ATOM:Cl]', '[ATOM:Cm]', '[ATOM:Co]', '[ATOM:Cr]', '[ATOM:Cs]', '[ATOM:Cu]',
                '[ATOM:Dy]', '[ATOM:Er]', '[ATOM:Es]', '[ATOM:Eu]', '[ATOM:F]', '[ATOM:Fe]', '[ATOM:Fm]', '[ATOM:Ga]', '[ATOM:Gd]', '[ATOM:Ge]',
                '[ATOM:He]', '[ATOM:Hf]', '[ATOM:Hg]', '[ATOM:Ho]', '[ATOM:I]', '[ATOM:In]', '[ATOM:Ir]', '[ATOM:K]', '[ATOM:Kr]', 
                '[ATOM:La]', '[ATOM:Li]', '[ATOM:Lr]', '[ATOM:Lu]', '[ATOM:Md]', '[ATOM:Mg]', '[ATOM:Mn]', '[ATOM:Mo]', 
                '[ATOM:N]', '[ATOM:Na]', '[ATOM:Nb]', '[ATOM:Nd]', '[ATOM:Ne]', '[ATOM:Ni]', '[ATOM:No]', '[ATOM:Np]', 
                '[ATOM:O]', '[ATOM:Os]', '[ATOM:P]', '[ATOM:Pa]', '[ATOM:Pb]', '[ATOM:Pd]', '[ATOM:Pm]', '[ATOM:Po]', '[ATOM:Pr]', '[ATOM:Pt]', '[ATOM:Pu]', 
                '[ATOM:Rb]', '[ATOM:Re]', '[ATOM:Rh]', '[ATOM:Rn]', '[ATOM:Ru]', 
                '[ATOM:S]', '[ATOM:Sb]', '[ATOM:Sc]', '[ATOM:Se]', '[ATOM:Si]', '[ATOM:Sm]', '[ATOM:Sn]', '[ATOM:Sr]', 
                '[ATOM:Ta]', '[ATOM:Tb]', '[ATOM:Tc]', '[ATOM:Te]', '[ATOM:Th]', '[ATOM:Ti]', '[ATOM:Tl]', '[ATOM:Tm]', 
                '[ATOM:U]', '[ATOM:V]', '[ATOM:W]', '[ATOM:Xe]', '[ATOM:Y]', '[ATOM:Yb]', '[ATOM:Zn]', '[ATOM:Zr]']
AtomPosition1 = [i[0:1]+"(DDD)"+i[1:] for i in AtomPosition]
AtomPosition2 = [i[0:1]+"(AAD)"+i[1:] for i in AtomPosition]
AtomPosition3 = [i[0:1]+"(XYZ)"+i[1:] for i in AtomPosition]


FPRTypes = ['[FPR:MACCS]', '[FPR:Toplogical]', '[FPR:ECFP]', '[FPR:FCFP]', '[FPR:Avalon]']

MathSymbols = ['0','1','2','3','4','5','6','7','8','9','.','-',';',',']

ControlSymbols = ['[sMLM_smi]','[sMLM_pro]','[sMLM_val]','[mMLM]',
                  '[sPLM_smi]','[sPLM_pro_val]','[mPLM]',
                  '[sPPLM_smi]','[sPPLM_pro_val]','[mPPLM]',
                  '[sGLM_smi]','[sGLM_pro]','[sGLM_val]','[mGLM_pro_val]',
                  '[sPGLM_smi]','[sPGLM_pro]','[sPGLM_val]','[mPGLM_smi_pro]','[mPGLM_smi_val]',
                  '[sPPM]','[mPPM]',

                  '[SPM_DDD]','[SPM_AAD]','[SPM_XYZ]']

ControlSymbolsPAIR = [i[:-1]+"(PAIR)"+i[-1:] for i in ControlSymbols]

                  # '[sMLM2_smi]','[sMLM2_pro]','[sMLM2_val]','[mMLM2]',
                  # '[sPLM2_smi]','[sPLM2_pro_val]','[mPLM2]',
                  # '[sPPLM2_smi]','[sPPLM2_pro_val]','[mPPLM2]',
                  # '[sGLM2_smi]','[sGLM2_pro]','[sGLM2_val]','[mGLM2_pro_val]',
                  # '[sPGLM2_smi]','[sPGLM2_pro]','[sPGLM2_val]','[mPGLM2_pro_val]','[mPGLM2_pro_val]','[mPGLM2_pro_val]']

from tokenizers import BertWordPieceTokenizer

tokenizer=BertWordPieceTokenizer(unk_token='[UNK]',sep_token='[SEP]',cls_token='[SOS]',pad_token='[PAD]',mask_token='[MASK]', lowercase=False)
tokenizer.train(['../vocab.txt'], special_tokens=['[PAD]','[SOS]','[EOS]','[UNK]','[MASK]'])
tokenizer.add_tokens(QuantumChemAttrs+LipinskiAttrs+DescriptorsAttrs+MathSymbols+ControlSymbols+ControlSymbolsPAIR+AtomPosition1+AtomPosition2+AtomPosition3+FPRTypes+['[SEP]','[SPAN]','[VALUE]','[XYZ]','[BITS]','[CUSPRO]','[SUB]'])
tokenizer.save('./tokenizer.json')
