import os
import sys
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from qmdesc import ReactivityDescriptorHandler
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import trange, tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from rdkit.Chem import AllChem
from matplotlib import pyplot as plt
import argparse

sys.path.append('..')

from Utils.TSEI import calc_TSEI

coupling = AllChem.ReactionFromSmarts('[P;+0:1].[P;+0:2].[Pd;+0:3]>>[P;+0:1]->[Pd;+0:3]<-[P;+0:2]')

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


solvent_dict = {'O': 'water', 'CCO': 'ethanol', 'CO': 'MeOH', 'C1CCNC1': 'THF', 'C1COCCO1': '1,4-dioxane',
                'CC#N': 'MeCN', 'Cc1ccccc1': 'toluene', 'COCCOC': '1,2-dimethoxyethane', 'CN(C)C=O': 'DMF',
                'CC(C)O': 'isopropyl alcohol',
                'CS(C)=O': 'dimethyl sulfoxide', 'ClCCl': 'DCM', 'CCOC(C)=O': 'EAC'}

dielectrics = {'water': 78.5, 'ethanol': 24.6, 'MeOH': 33, '1,4-dioxane': 2.2, 'toluene': 2.4, 'DMF': 37,
               'N,N-dimethyl-formamide': 37, 'tetrahydrofuran': 7.6, 'DCM': 9.08, 'EAC': 6.02,
               'THF': 7.6, 'MeCN': 38, 'isopropyl alcohol': 18.3, '1,2-dimethoxyethane': 3.5, 'acetonitrile': 38,
               'dimethyl sulfoxide': 49}
dipoles = {'water': 6.17, 'ethanol': 5.63, 'MeOH': 5.7, '1,4-dioxane': 1.5, 'toluene': 1.23, 'DMF': 12.73,
           'N,N-dimethyl-formamide': 12.73, 'tetrahydrofuran': 5.77, 'DCM': 5.34, 'EAC': 5.94,
           'THF': 5.77, 'MeCN': 13.07, 'isopropyl alcohol': 5.6, '1,2-dimethoxyethane': 5.8, 'acetonitrile': 13.07,
           'dimethyl sulfoxide': 11.57}

split_num = 4032

def get_qmdesc_SMC(x):
    print(x)
    h_sml, b_sml, ligand, reagent, solvents = x['h_mol'], x['b_mol'], x['ligand'], x['reagent'], x['solvent']
    h_mol, b_mol, ligand_mols, r_mols = Chem.MolFromSmiles(h_sml), Chem.MolFromSmiles(b_sml), [Chem.MolFromSmiles(l) for
                                                                                               l in
                                                                                               ligand.split(
                                                                                                   '.')], [
                                            Chem.MolFromSmiles(r) for r in reagent.split(',')]
    infos = []
    for idx, sml in enumerate([h_sml, b_sml]):
        handler = ReactivityDescriptorHandler()
        infos.append(handler.predict(sml))
    tseis = []
    for idx, mol in enumerate([h_mol, b_mol]):
        tseis.append(
            [calc_TSEI(mol, [i for i in range(len(mol.GetAtoms())) if i != j], j) for j in range(len(mol.GetAtoms()))])
    for i in range(len(infos)):
        infos[i]['tsei'] = np.array(tseis[i])

    l_infos = []
    for idx, sml in enumerate(ligand.split('.')):
        handler = ReactivityDescriptorHandler()
        l_infos.append(handler.predict(sml))
    l_tseis = []
    for idx, mol in enumerate(ligand_mols):
        l_tseis.append(
            [calc_TSEI(mol, [i for i in range(len(mol.GetAtoms())) if i != j], j) for j in range(len(mol.GetAtoms()))])
    for i in range(len(l_infos)):
        l_infos[i]['tsei'] = np.array(l_tseis[i])

    r_infos = []
    for idx, sml in enumerate(reagent.split(',')):
        handler = ReactivityDescriptorHandler()
        r_infos.append(handler.predict(sml))
    r_tseis = []
    for idx, mol in enumerate(r_mols):
        r_tseis.append(
            [calc_TSEI(mol, [i for i in range(len(mol.GetAtoms())) if i != j], j) for j in range(len(mol.GetAtoms()))])
    for i in range(len(r_infos)):
        r_infos[i]['tsei'] = np.array(r_tseis[i])

    descriptor = []

    h_num, h_C_num = -1, -1
    for atom in h_mol.GetAtoms():
        if atom.GetAtomicNum() in [9, 17, 35, 53]:
            for a in atom.GetNeighbors():
                if a.GetAtomicNum() == 6 and a.IsInRing() and a.GetIsAromatic():
                    h_C_num = a.GetIdx()
                    h_num = atom.GetIdx()
    if h_num == -1:
        for atom in h_mol.GetAtoms():
            if atom.GetAtomicNum() == 8 and 6 in [a.GetAtomicNum() for a in atom.GetNeighbors()] and 16 in [
                a.GetAtomicNum() for a in atom.GetNeighbors()]:
                for a in atom.GetNeighbors():
                    if a.GetAtomicNum() == 6 and a.IsInRing() and a.GetIsAromatic():
                        h_C_num = a.GetIdx()
                        h_num = atom.GetIdx()
    if h_num < 0 or h_C_num < 0:
        print(x['h_mol'])
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

    l_descriptor = []
    for l_idx, ligand_mol in enumerate(ligand_mols):
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
        l_descriptor.append(
            np.mean(np.array([l_infos[l_idx][j][ligand_nums] for j in (list(infos[0].keys())[1:5] + ['tsei'])]),
                    axis=1).tolist() + np.mean(
                np.array([l_infos[l_idx][j][ligand_nei_nums] for j in (list(infos[0].keys())[1:5] + ['tsei'])]),
                axis=1).tolist() + np.mean(
                np.array([l_infos[l_idx][j][ligand_bond_nums] for j in list(infos[0].keys())[5:7]]),
                axis=1).tolist())
    descriptor += np.mean(np.array(l_descriptor), axis=0).tolist()

    base_descriptor = []
    for r_idx, r_mol in enumerate(r_mols):
        base_nums = []
        for atom in r_mol.GetAtoms():
            if atom.GetAtomicNum() in [7, 8, 9] and atom.GetFormalCharge() < 0:
                base_nums.append(atom.GetIdx())
        if len(base_nums) == 0:
            for atom in r_mol.GetAtoms():
                if atom.GetAtomicNum() in [7, 8, 9]:
                    base_nums.append(atom.GetIdx())
        if len(base_nums) == 0:
            for atom in r_mol.GetAtoms():
                if atom.GetFormalCharge() < 0:
                    base_nums.append(atom.GetIdx())
        base_descriptor.append(
            np.mean(np.array([r_infos[r_idx][j][base_nums] for j in (list(infos[0].keys())[1:5] + ['tsei'])]),
                    axis=1).tolist())
    descriptor += np.mean(np.array(base_descriptor), axis=0).tolist()

    solvent_list = solvents.split('; ')
    solvent_name = solvent_list
    descriptor += [sum([dielectrics[s] for s in solvent_name]) / len(solvent_name),
                   sum([dipoles[s] for s in solvent_name]) / len(solvent_name)]

    return np.array(descriptor)


def gen_SMC_input():
    df = pd.read_csv('datasets/Reaxys_SMC.csv')
    dscp = df[['h_mol', 'b_mol', 'ligand', 'reagent', 'solvent']].apply(lambda x: get_qmdesc_SMC(x), axis=1)
    dscp = np.array(dscp)
    np.save(f'saved_descriptors/SMC_dscp_Reaxys.npy', dscp)


def classifier_yield_RF(num_fold=5, model_name='Reaxys.model', recalc=False):
    analyzed = pd.read_excel('../Utils/analyzed_descriptor.xlsx').SMC_R2_Reaxys.values
    if not os.path.exists('saved_descriptors/SMC_dscp_Reaxys.npy') or recalc:
        gen_SMC_input()
    X = np.load('saved_descriptors/SMC_dscp_Reaxys.npy', allow_pickle=True)
    Y = np.array(pd.read_csv('datasets/Reaxys_SMC.csv').rxn_yield.values)
    X = np.array([x.reshape(-1) for x in X])
    X = np.array([np.array(x) for x in X])[:, np.argsort(analyzed)[:9]]
    R2, RMSE = [], []
    max_R2 = -1
    MAE = []
    kf = KFold(n_splits=num_fold, shuffle=True)
    for num, (train_index, test_index) in enumerate(tqdm(kf.split(X, Y))):
        # X_train, X_test = X[train_index], X[test_index]
        # y_train, y_test = Y[train_index], Y[test_index]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2022 + num)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=2202 + num)
        model = RandomForestRegressor(oob_score=True, n_estimators=500)
        X_train = np.concatenate([X_train, X_val], axis=0)
        y_train = np.concatenate([y_train, y_val], axis=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        R2.append(r2_score(y_test, y_pred))
        RMSE.append(mean_squared_error(y_test, y_pred) ** 0.5)
        MAE.append(mean_absolute_error(y_test, y_pred))
        print(R2[-1], RMSE[-1], MAE[-1])
        if max_R2 < r2_score(y_test, y_pred):
            joblib.dump(model, f'saved_models/{model_name}')
            max_R2 = r2_score(y_test, y_pred)
    print(f'There are prediction outputs after {num_fold}-times training based on RandomForest:')
    # print('RandomForest' + ':', '%.4f' % np.array(Acc).mean() + ' ± ' + '%.4f' % np.array(Acc).std())
    print('RandomForest' + '|  R2_score:', '%.4f' % np.array(R2).mean() + ' ± ' + '%.4f' % np.array(R2).std(),
          ' RMSE:', '%.2f' % np.array(RMSE).mean() + ' ± ' + '%.2f' % np.array(RMSE).std(),
          ' MAE:', '%.2f' % np.array(MAE).mean() + ' ± ' + '%.2f' % np.array(MAE).std())
    print(f'Best Model has been saved as {model_name} in corresponding folder.\n')

classifier_yield_RF()

