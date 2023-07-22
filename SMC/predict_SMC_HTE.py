import pandas as pd
import numpy as np
from rdkit import Chem
from qmdesc import ReactivityDescriptorHandler
import os
import joblib
import sys
from tqdm import trange
from xgboost import XGBRegressor as XGBR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from rdkit.Chem import AllChem
import argparse

sys.path.append('..')

from Utils.TSEI import calc_TSEI

parser = argparse.ArgumentParser(description='Reaxtica: Predicting ddG for Chiral Phosphate Asymmetric Catalysis')
parser.add_argument('-m', '--mode', default='predict', choices=['predict', 'predict_file', 'retrain'],
                    help='Select whether to retrain the model or utilize existed model to predict.')
parser.add_argument('-rxn', '--rxn_sml', default='', help='The reaction/reaction file you intend to predict.')
parser.add_argument('--times_retrain', default=5)
parser.add_argument('--model_name', default='SMC_best.model', help='Set the name of your trained model.')
parser.add_argument('--save_name', default='predict_output.csv', help='Set the name of your predict output.')
parser.add_argument('--recalc', default=False, help='Whether to recalculate the descriptors.')
parser.add_argument('-dataset', '--dataset', default='', help='The dataset you are going to train or predict.')
args = parser.parse_args()

rxn_templates = [
    AllChem.ReactionFromSmarts('[c;+0:1]-[F;+0:2].[c;+0:3]-[B;+0:4]>>[c;+0:3]-[c;+0:1]'),
    AllChem.ReactionFromSmarts('[c;+0:1]-[Cl;+0:2].[c;+0:3]-[B;+0:4]>>[c;+0:3]-[c;+0:1]'),
    AllChem.ReactionFromSmarts('[c;+0:1]-[Br;+0:2].[c;+0:3]-[B;+0:4]>>[c;+0:3]-[c;+0:1]'),
    AllChem.ReactionFromSmarts('[c;+0:1]-[I;+0:2].[c;+0:3]-[B;+0:4]>>[c;+0:3]-[c;+0:1]')]

dielectrics = {'water': 78.5, 'ethanol': 24.6, 'MeOH': 33, '1,4-dioxane': 2.2, 'toluene': 2.4, 'DMF': 37,
               'THF': 7.6, 'MeCN': 38, 'isopropyl alcohol': 18.3, '1,2-dimethoxyethane': 3.5,
               'dimethyl sulfoxide': 49}
dipoles = {'water': 6.17, 'ethanol': 5.63, 'MeOH': 5.7, '1,4-dioxane': 1.5, 'toluene': 1.23, 'DMF': 12.73,
           'THF': 5.77, 'MeCN': 13.07, 'isopropyl alcohol': 5.6, '1,2-dimethoxyethane': 5.8,
           'dimethyl sulfoxide': 11.57}

split_num = 4032

def get_qmdesc_SMC(x):
    h_sml, b_sml, ligand, reagent, solvent = x['h_mol'], x['b_mol'], x['ligand'], x['reagent'], x['solvent']
    h_mol, b_mol, ligand_mol, r_mol = Chem.MolFromSmiles(h_sml), Chem.MolFromSmiles(b_sml), Chem.MolFromSmiles(ligand), Chem.MolFromSmiles(reagent)
    infos = []
    for idx, sml in enumerate([h_sml, b_sml, ligand, reagent]):
        clean_sml = sml.replace('/', '%').replace('\\', '$').replace(':', '~')
        if not os.path.exists(f'../qmdesc_bank/{clean_sml}.npy'):
            handler = ReactivityDescriptorHandler()
            infos.append(handler.predict(sml))
            np.save(f'../qmdesc_bank/{clean_sml}.npy', infos[idx])
        else:
            infos.append(np.load(f'../qmdesc_bank/{clean_sml}.npy', allow_pickle=True).item())
    tseis = []
    for idx, mol in enumerate([h_mol, b_mol, ligand_mol, r_mol]):
        tseis.append(
            [calc_TSEI(mol, [i for i in range(len(mol.GetAtoms())) if i != j], j) for j in range(len(mol.GetAtoms()))])
    for i in range(len(infos)):
        infos[i]['tsei'] = np.array(tseis[i])

    descriptor = []

    h_num, h_C_num = -1, -1
    for atom in h_mol.GetAtoms():
        if atom.GetAtomicNum() in [8, 17, 35, 53]:
            for a in atom.GetNeighbors():
                if a.GetAtomicNum() == 6 and a.IsInRing() and a.GetIsAromatic():
                    h_C_num = a.GetIdx()
                    h_num = atom.GetIdx()
    h_bond_num = h_mol.GetBondBetweenAtoms(h_num, h_C_num).GetIdx()
    descriptor += ([infos[0][j][h_num] for j in list(infos[0].keys())[1:5]] + [tseis[0][h_num]])
    descriptor += ([infos[0][j][h_C_num] for j in list(infos[0].keys())[1:5]] + [tseis[0][h_C_num]])
    descriptor += [infos[0][j][h_bond_num] for j in list(infos[0].keys())[5:7]]

    b_num, b_nei_num = -1, -1
    for atom in b_mol.GetAtoms():
        if atom.GetAtomicNum() == 5:
            b_num = atom.GetIdx()
    for atom in b_mol.GetAtoms()[b_num].GetNeighbors():
        if atom.GetIsAromatic() and atom.GetAtomicNum() == 6 and atom.IsInRing():
            b_nei_num = atom.GetIdx()
    b_bond_num = b_mol.GetBondBetweenAtoms(b_num, b_nei_num).GetIdx()
    descriptor += ([infos[1][j][b_num] for j in list(infos[1].keys())[1:5]] + [tseis[1][b_num]])
    descriptor += ([infos[1][j][b_nei_num] for j in list(infos[1].keys())[1:5]] + [tseis[1][b_nei_num]])
    descriptor += [infos[1][j][b_bond_num] for j in list(infos[1].keys())[5:7]]

    ligand_nums, ligand_nei_nums, ligand_bond_nums = [], [], []
    for atom in ligand_mol.GetAtoms():
        if atom.GetAtomicNum() == 15:
            ligand_nums.append(atom.GetIdx())
    if len(ligand_nums) == 0:
        for atom in ligand_mol.GetAtoms():
            if atom.GetAtomicNum() == 8:
                ligand_nums.append(atom.GetIdx())
    for num in ligand_nums:
        for a in ligand_mol.GetAtoms()[num].GetNeighbors():
            ligand_nei_nums.append(a.GetIdx())
        for b in ligand_mol.GetAtoms()[num].GetBonds():
            ligand_bond_nums.append(b.GetIdx())
    ligand_nei_nums, ligand_bond_nums = list(set(ligand_nei_nums)), list(set(ligand_bond_nums))
    descriptor += np.mean(np.array([infos[2][j][ligand_nums] for j in (list(infos[0].keys())[1:5] + ['tsei'])]),
                          axis=1).tolist()
    descriptor += np.mean(np.array([infos[2][j][ligand_nei_nums] for j in (list(infos[0].keys())[1:5] + ['tsei'])]),
                          axis=1).tolist()
    descriptor += np.mean(np.array([infos[2][j][ligand_bond_nums] for j in list(infos[0].keys())[5:7]]),
                          axis=1).tolist()

    base_nums = []
    for atom in r_mol.GetAtoms():
        if atom.GetAtomicNum() in [7, 8, 9]:
            base_nums.append(atom.GetIdx())
    descriptor += np.mean(np.array([infos[3][j][base_nums] for j in (list(infos[0].keys())[1:5] + ['tsei'])]),
                          axis=1).tolist()

    descriptor += [dielectrics[solvent], dipoles[solvent]]

    return np.array(descriptor)


def gen_SMC_input():
    df = pd.read_csv('datasets/HTE_SMC.csv')
    dscp = df[['h_mol', 'b_mol', 'ligand', 'reagent', 'solvent']].apply(lambda x: get_qmdesc_SMC(x), axis=1)
    dscp = np.array(dscp)
    np.save(f'saved_descriptors/SMC_dscp_HTE.npy', dscp)



def classifier_yield_XGB(num_fold=10, model_name='SMC_XGB.model', recalc=False):
    if not os.path.exists(f'saved_descriptors/SMC_dscp_HTE.npy') or recalc:
        gen_SMC_input()
    X = np.load(f'saved_descriptors/SMC_dscp_HTE.npy', allow_pickle=True)
    analyzed = pd.read_excel('../Utils/analyzed_descriptor.xlsx').SMC_R2_HTE.values
    Y = pd.read_csv('datasets/HTE_SMC.csv').rxn_yield.values
    X = np.array([x.reshape(-1) for x in X])
    orig_X = X
    X = orig_X[:, analyzed.argsort()[:42]]
    R2, RMSE = [], []
    max_R2 = -1
    MAE = []
    for num in trange(num_fold):
        split = pd.read_excel(f'random_splits/random_split_{num}.xlsx')['Unnamed: 0'].values
        X_train, X_test = X[split[:split_num]], X[split[split_num:]]
        y_train, y_test = Y[split[:split_num]], Y[split[split_num:]]
        model = XGBR(n_estimators=200, max_depth=100, gamma=0.03, reg_alpha=25, reg_lambda=25)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        R2.append(r2_score(y_test, y_pred))
        RMSE.append(mean_squared_error(y_test, y_pred) ** 0.5)
        MAE.append(mean_absolute_error(y_test, y_pred))
        if max_R2 < r2_score(y_test, y_pred):
            joblib.dump(model, f'saved_models/SMC_HTE_{model_name}')
            max_R2 = r2_score(y_test, y_pred)
    print(f'There are prediction outputs after {num_fold}-times training based on RandomForest:')
    print('RandomForest' + '|  R2_score:', '%.4f' % np.array(R2).mean() + ' ± ' + '%.4f' % np.array(R2).std(),
          ' RMSE:', '%.2f' % np.array(RMSE).mean() + ' ± ' + '%.2f' % np.array(RMSE).std(),
          ' MAE:', '%.3f' % np.array(MAE).mean() + ' ± ' + '%.3f' % np.array(MAE).std())
    print(f'Best Model has been saved as {model_name} in corresponding folder.\n')

if args.mode == 'retrain':
    classifier_yield_XGB()




