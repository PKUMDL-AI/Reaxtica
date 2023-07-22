import os
import sys
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from qmdesc import ReactivityDescriptorHandler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import trange
import argparse

sys.path.append('..')

from Utils.TSEI import calc_TSEI

parser = argparse.ArgumentParser(description='Reaxtica: Predicting ddG for Chiral Phosphate Asymmetric Catalysis')
parser.add_argument('-m', '--mode', default='predict', choices=['predict', 'predict_file', 'retrain'],
                    help='Select whether to retrain the model or utilize existed model to predict.')
parser.add_argument('-rxn', '--rxn_sml', default='', help='The reaction/reaction file you intend to predict.')
parser.add_argument('--model_name', default='BHC_best.model', help='Set the name of your trained model.')
parser.add_argument('--times_retrain', default=5)
parser.add_argument('--save_name', default='predict_output.csv', help='Set the name of your predict output.')
parser.add_argument('--recalc', default=False, help='Whether to recalculate the descriptors.')
parser.add_argument('-dataset', '--dataset', default='', help='The dataset you are going to train or predict.')
args = parser.parse_args()


def get_qmdesc_BHC(x): # generate descriptors for BHC
    try:
        ligand, additive, base, halide = Chem.MolFromSmiles(x['Ligand']), Chem.MolFromSmiles(
            x['Additive']), Chem.MolFromSmiles(x['Base']), Chem.MolFromSmiles(x['Aryl halide'])
        l_sml, a_sml, b_sml, h_sml = x['Ligand'], x['Additive'], x['Base'], x['Aryl halide']
    except:
        l_sml, a_sml, b_sml, h_sml = x.split('.')[0], x.split('.')[1], x.split('.')[2], x.split('.')[3]
        ligand, additive, base, halide = Chem.MolFromSmiles(l_sml), Chem.MolFromSmiles(a_sml), Chem.MolFromSmiles(b_sml), Chem.MolFromSmiles(h_sml)
    infos = []
    for idx, sml in enumerate([l_sml, a_sml, b_sml, h_sml]):
        handler = ReactivityDescriptorHandler()
        infos.append(handler.predict(sml))
    tseis = []
    for idx, mol in enumerate([ligand, additive, base, halide]):
        tseis.append(
            [calc_TSEI(mol, [i for i in range(len(mol.GetAtoms())) if i != j], j) for j in range(len(mol.GetAtoms()))])
    descriptor = []

    for atom in ligand.GetAtoms():
        if atom.GetAtomicNum() == 15: # P
            ligand_P_num = atom.GetIdx()
    descriptor += ([infos[0][j][ligand_P_num] for j in list(infos[0].keys())[1:4]] + [
        infos[0]['NMR'][ligand_P_num]] + [tseis[0][ligand_P_num]])
    additive_N_num, additive_O_num, additive_C_num = -1, -1, -1
    for atom in additive.GetAtoms():
        if atom.GetAtomicNum() == 7 and atom.GetIsAromatic() and atom.IsInRing() and atom.GetDegree() == 2:
            additive_N_num = atom.GetIdx()
            for neigh_atom in atom.GetNeighbors():
                if neigh_atom.GetAtomicNum() == 8:
                    additive_O_num = neigh_atom.GetIdx()
                if neigh_atom.GetAtomicNum() == 6:
                    additive_C_num = neigh_atom.GetIdx()
            assert min([additive_N_num, additive_O_num, additive_C_num]) >= 0
            break
    additive_bond_num = additive.GetBondBetweenAtoms(additive_N_num, additive_O_num).GetIdx()
    for additive_num in [additive_N_num, additive_O_num, additive_C_num]:
        descriptor += ([infos[1][j][additive_num] for j in list(infos[1].keys())[1:4]] + [
            infos[1]['NMR'][additive_num]] + [tseis[1][additive_num]])
    descriptor += [infos[1][j][additive_bond_num] for j in list(infos[1].keys())[5:7]]

    base_N_nums = []
    for atom in base.GetAtoms():
        if atom.GetAtomicNum() == 7: # N
            base_N_nums.append(atom.GetIdx())
    base_info = [[infos[2][j][base_N_num] for j in list(infos[2].keys())[1:4]] + [
        infos[2]['NMR'][base_N_num]] + [tseis[2][base_N_num]] for base_N_num in base_N_nums]
    base_info = np.array(base_info).mean(axis=0).tolist()
    descriptor += base_info

    for atom in halide.GetAtoms():
        if atom.GetAtomicNum() in [17, 35, 53]: # Cl, Br, I
            halide_X_num = atom.GetIdx()
            halide_C_num = atom.GetNeighbors()[0].GetIdx()
            halide_bond_num = atom.GetBonds()[0].GetIdx()
        break
    for halide_num in [halide_X_num, halide_C_num]:
        descriptor += ([infos[3][j][halide_num] for j in list(infos[3].keys())[1:4]] + [
            infos[3]['NMR'][halide_num]] + [tseis[3][halide_num]])
    descriptor += [infos[3][j][halide_bond_num] for j in list(infos[3].keys())[5:7]]

    return np.array(descriptor)


def gen_BHC_input(sheetname):
    df = pd.read_excel('datasets/BHC_input.xlsx', sheet_name=sheetname)
    dscp = df[['Ligand', 'Additive', 'Base', 'Aryl halide']].apply(lambda x: get_qmdesc_BHC(x), axis=1)
    dscp = np.array(dscp)
    np.save(f'saved_descriptors/{sheetname}.npy', dscp)


def classifier_yield_RF(sheetname, num_fold=5, model_name='best.model', recalc=False): # generate model for BHC
    analyzed = pd.read_excel('../Utils/analyzed_descriptor.xlsx').BHC_delta_R_square.values
    if sheetname == 'random':
        split_num = 2767
        average_R2, average_RMSE, average_MAE = [], [], []
        for num in range(1, 10):
            sheet = f'FullCV_0{num}'
            if not os.path.exists(f'saved_descriptors/{sheet}.npy') or recalc:
                gen_BHC_input(sheet)
            X = np.load(f'saved_descriptors/{sheet}.npy', allow_pickle=True)
            Y = pd.read_excel('datasets/BHC_input.xlsx', sheet_name=sheet).Output.values
            X = np.array([x.reshape(-1) for x in X])[:, np.argsort(np.array(analyzed))[:13]]
            R2, RMSE, MAE = [], [], []
            max_R2 = 0
            for _ in trange(num_fold):
                X_train, X_test, y_train, y_test = X[:split_num], X[split_num:], Y[:split_num], Y[split_num:]
                model = XGBRegressor()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                R2.append(r2_score(y_test, y_pred))
                RMSE.append(mean_squared_error(y_test, y_pred) ** 0.5)
                MAE.append(mean_absolute_error(y_test, y_pred))
                if max_R2 < r2_score(y_test, y_pred):
                    joblib.dump(model, f'saved_models/{sheet}_{model_name}')
                    max_R2 = r2_score(y_test, y_pred)
            print(f'Random Validation for BHC: {num}')
            print('RandomForest' + '|  R2_score:', '%.4f' % np.array(R2).mean() + ' ± ' + '%.4f' % np.array(R2).std(),
                  ' RMSE:', '%.3f' % np.array(RMSE).mean() + ' ± ' + '%.3f' % np.array(RMSE).std(),
                  ' MAE:', '%.3f' % np.array(MAE).mean() + ' ± ' + '%.3f' % np.array(MAE).std())
            average_R2.append(np.array(R2).mean())
            average_RMSE.append(np.array(RMSE).mean())
            average_MAE.append(np.array(MAE).mean())
        sheet = 'FullCV_10'
        if not os.path.exists(f'saved_descriptors/{sheet}.npy') or recalc:
            gen_BHC_input(sheet)
        X = np.load(f'saved_descriptors/{sheet}.npy', allow_pickle=True)
        Y = pd.read_excel('datasets/BHC_input.xlsx', sheet_name=sheet).Output.values
        X = np.array([x.reshape(-1) for x in X])[:, np.argsort(np.array(analyzed))[:13]]
        R2, RMSE, MAE = [], [], []
        max_R2 = 0
        for _ in trange(num_fold):
            X_train, X_test, y_train, y_test = X[:split_num], X[split_num:], Y[:split_num], Y[split_num:]
            model = XGBRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            R2.append(r2_score(y_test, y_pred))
            RMSE.append(mean_squared_error(y_test, y_pred) ** 0.5)
            MAE.append(mean_absolute_error(y_test, y_pred))
            if max_R2 < r2_score(y_test, y_pred):
                joblib.dump(model, f'saved_models/{sheet}_{model_name}')
                max_R2 = r2_score(y_test, y_pred)
        print(f'Random Validation for BHC: 10')
        print('RandomForest' + '|  R2_score:', '%.4f' % np.array(R2).mean() + ' ± ' + '%.4f' % np.array(R2).std(),
              ' RMSE:', '%.3f' % np.array(RMSE).mean() + ' ± ' + '%.3f' % np.array(RMSE).std(),
              ' MAE:', '%.3f' % np.array(MAE).mean() + ' ± ' + '%.3f' % np.array(MAE).std())
        average_R2.append(np.array(R2).mean())
        average_RMSE.append(np.array(RMSE).mean())
        average_MAE.append(np.array(MAE).mean())
        print('Average on 10 random validation for BHC')
        print('RandomForest' + '|  R2_score:',
              '%.4f' % np.array(average_R2).mean() + ' ± ' + '%.4f' % np.array(average_R2).std(),
              ' RMSE:', '%.3f' % np.array(average_RMSE).mean() + ' ± ' + '%.3f' % np.array(average_RMSE).std(),
              ' MAE:', '%.3f' % np.array(average_MAE).mean() + ' ± ' + '%.3f' % np.array(average_MAE).std())
    else:
        if 'FullCV' in sheetname:
            split_num = 2767
        elif '1' in sheetname:
            split_num = 3057
        elif '3' in sheetname:
            split_num = 3058
        else:
            split_num = 3055
        if not os.path.exists(f'saved_descriptors/{sheetname}.npy') or recalc:
            gen_BHC_input(sheetname)
        X = np.load(f'saved_descriptors/{sheetname}.npy', allow_pickle=True)
        Y = pd.read_excel('datasets/BHC_input.xlsx', sheet_name=sheetname).Output.values
        X = np.array([x.reshape(-1) for x in X])[:, np.argsort(np.array(analyzed))[:13]]
        R2, RMSE, MAE = [], [], []
        max_R2 = 0
        for _ in trange(num_fold):
            X_train, X_test, y_train, y_test = X[:split_num], X[split_num:], Y[:split_num], Y[split_num:]
            model = XGBRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            R2.append(r2_score(y_test, y_pred))
            RMSE.append(mean_squared_error(y_test, y_pred) ** 0.5)
            MAE.append(mean_absolute_error(y_test, y_pred))
            if max_R2 < r2_score(y_test, y_pred):
                joblib.dump(model, f'saved_models/{model_name}')
                max_R2 = r2_score(y_test, y_pred)
        print(f'There are prediction outputs after {num_fold}-times training based on RandomForest.')
        print('RandomForest' + '|  R2_score:', '%.4f' % np.array(R2).mean() + ' ± ' + '%.4f' % np.array(R2).std(),
              ' RMSE:', '%.3f' % np.array(RMSE).mean() + ' ± ' + '%.3f' % np.array(RMSE).std(),
              ' MAE:', '%.3f' % np.array(MAE).mean() + ' ± ' + '%.3f' % np.array(MAE).std())
        print(f'Best Model has been saved as {model_name} in corresponding folder.\n')


def predict_single_BHC(x, model_name='best.model'):
    try:
        analyzed = pd.read_excel('../Utils/analyzed_descriptor.xlsx').BHC_delta_R_square.values
        dscp = get_qmdesc_BHC(x)[np.argsort(np.array(analyzed))[:13]]
        model = joblib.load(f'saved_models/{model_name}')
        yield_pred = model.predict(dscp.reshape(1, -1))
        print(f'The yield of this reaction is predicted to be {round(yield_pred.item(), 2)}%.')
    except:
        print('There\'s something wrong in your input, please check again.')


def predict_file_BHC(filepath, model_name='best.model', save_name='predict_output.csv'):
    analyzed = pd.read_excel('../Utils/analyzed_descriptor.xlsx').BHC_delta_R_square.values
    df = pd.read_csv(f'datasets/{filepath}')
    dscp = df[['Ligand', 'Additive', 'Base', 'Aryl halide']].apply(lambda x: get_qmdesc_BHC(x), axis=1)
    dscp = np.array([x.reshape(-1) for x in dscp])[:, np.argsort(np.array(analyzed))[:13]]
    model = joblib.load(f'saved_models/{model_name}')
    yield_pred = model.predict(dscp)
    df['yield'] = yield_pred
    df.to_csv(f'datasets/{save_name}', index=None)
    print(f'Reaction yield prediction has been saved as {save_name} in corresponding folder.\n')

if args.mode == 'retrain':
    classifier_yield_RF(args.dataset, args.times_retrain, args.model_name, args.recalc)
if args.mode == 'predict':
    BHC_yield = predict_single_BHC(args.rxn_sml, args.model_name)
if args.mode == 'predict_file':
    predict_file_BHC(args.rxn_sml, args.model_name, args.save_name)