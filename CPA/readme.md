# CPA

```
cd CPA
# retrain random validation
python predict_CPA.py -m retrain -dataset random
# retrain test
python predict_CPA.py -m retrain -dataset test_sub(or test_cat, test_sub-cat)
# predict single reaction
python predict_CPA.py -rxn 'O=P1(O)OC2=C(C3=CC=CC=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC=CC=C6)=CC7=C5C=CC=C7.O=C(C1=CC=CC=C1)/N=C/C2=CC=C(C(F)(F)F)C=C2.SC1=CC=CC=C1C'
# (order := $ligand.$imine.$thiol)
```