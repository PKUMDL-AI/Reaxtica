# BHC

```
cd BHC
# retrain random validation
python predict_BHC.py -m retrain -dataset random
# retrain test
python predict_BHC.py -m retrain -dataset Test1(or Test2, Test3, Test4)
# predict single reaction
python predict_BHC.py --rxn 'CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P([C@@]3(C[C@@H]4C5)C[C@H](C4)C[C@H]5C3)[C@]6(C7)C[C@@H](C[C@@H]7C8)C[C@@H]8C6)C(OC)=CC=C2OC.CC1=CC(C)=NO1.CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC.ClC1=NC=CC=C1'
# (order := $ligand.$additive.$base.$halide)
```
