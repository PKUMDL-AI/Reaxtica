# DA

```
cd DA
# retrain the regioselectivity
python predict_DA.py -m retrain -dataset DA_input_regio.csv
# retrain the regio- & site-selectivity
python predict_DA.py -m retrain -dataset DA_input_all.csv
# predict the external file
python predict_DA.py -m predict_file -rxn DA_19_all.csv
# predict single reaction
python predict_DA.py -rxn 'COC(=O)N(CC1=CC=CC=C1)C=CC=C.C=CC=O' --temp 20 --acid 1
# (the value of temperature is set as 20 while the acid is set as None as default if they are not typed in)
```