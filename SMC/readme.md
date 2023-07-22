
# SMC

```
cd SMC
# retrain random validation for USPTO
python predict_SMC_USPTO.py -m retrain
# retrain random validation for HTE
python predict_SMC_HTE.py -m retrain
# predict single reaction
python predict_SMC_USPTO.py -rxn 'Br[c:8]1[c:3]([C:1]#[N:2])[n:4][cH:5][cH:6][cH:7]1,OB(O)[c:13]1[cH:12][cH:11][c:10]([CH3:19])[cH:15][cH:14]1,c1ccc(P(c2ccccc2)c2ccccc2)cc1,O=C([O-])[O-].[Na+].[Na+],CCO.Cc1ccccc1'
# (order := $halide,$boric_acid,$ligand,$base,$solvent)
# Note that comma is used instead of dot for seperation.
# Or
python predict_SMC_USPTO.py -rxn 'Br[c:2]1[cH:3][n:4][cH:5][c:6]([Br:8])[cH:7]1;CCB(CC)[c:13]1[cH:12][n:11][cH:16][cH:15][cH:14]1;c1ccc(P(c2ccccc2)c2ccccc2)cc1;[K+].[OH-],[Cl-].[Na+],O=S(=O)([O-])[O-].[Mg+2];CCOC(C)=O'
# (order := $halide;$boric_acid;$ligand;$base1,$base2,...;$solvent peration)
# Semicolon is used as the seperator while there is more than one kind of bases.
```