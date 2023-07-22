import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from qmdesc import ReactivityDescriptorHandler
import time
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
from tqdm import trange, tqdm

sys.path.append('..')

from Utils.TSEI import calc_TSEI
from Utils.mol_utils import extract_substrate, GetAtomWithMapNumber, retro_DA_SMARTS, extract_substituent
from warnings import filterwarnings
import argparse

filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Reaxtica: Predicting selectivity for Diels-Alder Cycloaddition')
parser.add_argument('-m', '--mode', default='predict', choices=['predict', 'predict_file', 'retrain'],
                    help='Select whether to retrain the model or utilize existed model to predict.')
parser.add_argument('-rxn', '--rxn_sml', default='', help='The reaction/reaction file you intend to predict.')
parser.add_argument('--model_name', default='all.model', help='Set the name of your trained model.')
parser.add_argument('--times_retrain', default=5)
parser.add_argument('--save_name', default='predict_output.csv', help='Set the name of your predict output.')
parser.add_argument('--recalc', default=False, help='Whether to recalculate the descriptors.')
parser.add_argument('--temp', default=20, help='The reaction temperature')
parser.add_argument('--acid', default=0, help='Whether the reaction is catalyzed by external lewis acid')
parser.add_argument('--DA_jobtype', default='all', choices=['regio', 'all'],
                    help='Select to predict regio-selectivity or regio&site-selectvity.')
parser.add_argument('-dataset', '--dataset', default='', help='The dataset you are going to train or predict.')
args = parser.parse_args()

def operate_DA(r1_sml, r2_sml): # running virtial DA reactions
    rxn = AllChem.ReactionFromSmarts(
        '[C;+0:1]=[C;+0:2].[C;+0:3]=[C;+0:4]-[C;+0:5]=[C;+0:6]>>[C;+0:3]1-[C;+0:1]-[C;+0:2]-[C;+0:6]-[C;+0:5]=[C;+0:4]-1')
    sigma_trans_subs = [Chem.MolFromSmiles(sub_sml) for sub_sml in
                        'C=C1CC=C1.C=C1CCC=C1.C=C1CCCC=C1.C=C1CCCCC=C1'.split('.')]
    sigma_trans_flag = False
    r1_mol, r2_mol = Chem.MolFromSmiles(r1_sml), Chem.MolFromSmiles(r2_sml)
    for idx, atom in enumerate(r1_mol.GetAtoms()):
        atom.SetAtomMapNum(idx + 1)
    for idx, atom in enumerate(r2_mol.GetAtoms()):
        atom.SetAtomMapNum(len(r1_mol.GetAtoms()) + idx + 1)
    for sub in sigma_trans_subs:
        if r1_mol.HasSubstructMatch(sub):
            sigma_trans_flag = True
            sigma_trans_idx = r1_mol.GetSubstructMatch(sub)
            sigma_trans_mapnum = [r1_mol.GetAtoms()[idx].GetAtomMapNum() for idx in sigma_trans_idx]
        if r2_mol.HasSubstructMatch(sub):
            sigma_trans_flag = True
            sigma_trans_idx = r2_mol.GetSubstructMatch(sub)
            sigma_trans_mapnum = [r2_mol.GetAtoms()[idx].GetAtomMapNum() for idx in sigma_trans_idx]
    r1_sml, r2_sml = Chem.MolToSmiles(r1_mol), Chem.MolToSmiles(r2_mol)
    products_run = [Chem.MolToSmiles(i[0]) for i in rxn.RunReactants((r1_mol, r2_mol))]
    if not products_run:
        products_run = [Chem.MolToSmiles(i[0]) for i in rxn.RunReactants((r2_mol, r1_mol))]
    products_run = list(set(products_run))
    if sigma_trans_flag:
        products_run = [product for product in products_run if
                        sum([int(mapnum not in [a.GetAtomMapNum() for a in Chem.MolFromSmiles(product).GetAtoms()]) for
                             mapnum in sigma_trans_mapnum]) < 4]
    products_inchi, clean_product = [], []
    for product in products_run:
        if Chem.MolToInchi(Chem.MolFromSmiles(product)) not in products_inchi:
            products_inchi.append(Chem.MolToInchi(Chem.MolFromSmiles(product)))
            clean_product.append(product)
    return clean_product, r1_sml, r2_sml

def get_qmdesc_feature(sml, re_calc=False):
    mol = Chem.MolFromSmiles(sml)
    handler = ReactivityDescriptorHandler()
    infos = handler.predict(sml)
    num_atoms = len(mol.GetAtoms())
    mol = Chem.MolFromSmiles(sml)
    tsei = [calc_TSEI(mol, [i for i in range(len(mol.GetAtoms())) if i != j], j) for j in range(len(mol.GetAtoms()))]
    mol = Chem.AddHs(mol)
    atom_feature = []
    for atom_id, atom in enumerate(list(mol.GetAtoms())[:num_atoms]):
        atom_feature.append(
            np.array([infos[j][atom_id] for j in list(infos.keys())[1:5]] + [
                tsei[atom_id]]))
    return np.array(atom_feature)


def is_regioisomer(sml1, sml2): # check whether two products are regioisomers
    mol1, mol2 = Chem.MolFromSmiles(sml1), Chem.MolFromSmiles(sml2)
    rxn = AllChem.ReactionFromSmarts(
        '[C;+0:3]1-[C;+0:1]-[C;+0:2]-[C;+0:6]-[C;+0:5]=[C;+0:4]-1>>[C;+0:1]=[C;+0:2].[C;+0:3]=[C;+0:4]-[C;+0:5]=['
        'C;+0:6]')
    retro_product1 = [(Chem.MolToInchi(i[0]), Chem.MolToInchi(i[1])) for i in rxn.RunReactants([mol1])]
    retro_product2 = [(Chem.MolToInchi(i[0]), Chem.MolToInchi(i[1])) for i in rxn.RunReactants([mol2])]
    for retro in retro_product1:
        if retro in retro_product2:
            return 1
    return 0


def get_DA_dscp(sml, products_run, jobtype, temp, acid):
    reactants = sml[:sml.find('>>')] if sml.find('>>') != -1 else sml
    rxn_vector, label = [], []
    s, p, r = extract_substrate(sml)
    p_mol = Chem.MolFromSmiles(p)
    rxn_order = retro_DA_SMARTS(sml)
    _, corr_subs = extract_substituent(sml, rxn_order)
    valid_p_num = len(products_run)
    for product in products_run:
        p_vector = []
        try:
            _, allsub = extract_substituent(f'{reactants}>>{product}', rxn_order)
        except AttributeError:
            _, allsub = extract_substituent(f'{reactants}>>{product}', abs(1 - rxn_order))
        for sub in allsub.split('.'):
            if sub == '[La][H]':
                p_vector.append(get_qmdesc_feature('[H][H]')[0])
            else:
                real_sml = sub.replace('[La]', '[H]')
                mol = Chem.MolFromSmiles(sub)
                real_mol = Chem.MolFromSmiles(real_sml)
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == 'La':
                        if len(atom.GetNeighbors()) == 1:
                            sub_map_id = atom.GetNeighbors()[0].GetIntProp('molAtomMapNumber')
                        else:
                            raise AssertionError
                raw_dscp = get_qmdesc_feature(real_sml)
                sub_atom_id = GetAtomWithMapNumber(real_mol, sub_map_id).GetIdx()
                p_vector.append(raw_dscp[sub_atom_id])
        p_vector = np.array(p_vector).reshape(-1)
        l = int(p_vector.shape[0] / 10)
        rxn_vector.append(p_vector)
        rxn_vector.append(np.concatenate(
            [p_vector[:l * 6], p_vector[l * 9:], p_vector[l * 8:l * 9], p_vector[l * 7:l * 8], p_vector[l * 6:l * 7]]))
        rxn_vector.append(np.concatenate(
            [p_vector[l * 5:l * 6], p_vector[l * 4:l * 5], p_vector[l * 3:l * 4], p_vector[l * 2:l * 3],
             p_vector[l:l * 2], p_vector[:l], p_vector[l * 6:]]))
        rxn_vector.append(np.concatenate(
            [p_vector[l * 5:l * 6], p_vector[l * 4:l * 5], p_vector[l * 3:l * 4], p_vector[l * 2:l * 3],
             p_vector[l:l * 2], p_vector[:l], p_vector[l * 9:], p_vector[l * 8:l * 9], p_vector[l * 7:l * 8],
             p_vector[l * 6:l * 7]]))
        product_mol = Chem.MolFromSmiles(product)
        if jobtype == 'regio':
            if p_mol.HasSubstructMatch(product_mol):
                label += [1, 0, 0, 1]
            else:
                label += [0, 1, 1, 0]
        elif jobtype == 'all':
            if p_mol.HasSubstructMatch(product_mol):
                label += [1, 0, 0, 1]
            elif is_regioisomer(product, p):
                label += [0, 1, 1, 0]
            else:
                label += [0, 0, 0, 0]
    rxn_vector = np.array(rxn_vector).reshape(valid_p_num * 4, -1)
    rxn_vector = np.concatenate([rxn_vector, np.array([temp for _ in range(rxn_vector.shape[0])]).reshape(-1, 1),
                                     np.array([acid for _ in range(rxn_vector.shape[0])]).reshape(-1, 1)], axis=1)
    return rxn_vector, label


def vectorize_DA(filename, jobtype):
    print(f'Constructing vectorized descriptors...\nYour chosen dataset is:{filename}\n')
    df = pd.read_csv(f'datasets/{filename}')
    rxn_vectors = []
    labels = []
    err_num = 0
    err_list = []
    for i in trange(df.shape[0]):
        sml = df['rxn_smiles'][i]
        possible_products = df['products_run'][i].split('.')
        rxn_vector, label = get_DA_dscp(sml, possible_products, jobtype, df['Temp'][i], df['Lewis_acid'][i])
        rxn_vectors.append(rxn_vector)
        labels.append(label)
    print('Illegal inputs:\n')
    print(np.array(err_list))
    print('Num of illegal input:', str(err_num) + ',', 'Percentage:', str('%.2f' % (100 * err_num / df.shape[0])) + '%')
    np.savez(f'saved_descriptors/{filename}_dscp.npz', np.array(rxn_vectors), np.array(labels))
    print('Vectorized descriptors are constructed.\n')
    return rxn_vectors, labels


def classifier_single_DA(num_fold=5, model_name='best.model', recalc=False, filename='DA_input_all.csv',
                         jobtype=''):
    if jobtype == '':
        jobtype = 'regio' if 'regio' in filename else ('all' if 'all' in filename else '')
    if jobtype == '':
        jobtype = 'regio'
    if os.path.exists(f'saved_descriptors/{filename}_dscp.npz') and not recalc:
        raw_file = np.load(f'saved_descriptors/{filename}_dscp.npz', allow_pickle=True)
        X, Y = raw_file['arr_0'], raw_file['arr_1']
    else:
        X, Y = vectorize_DA(filename, jobtype)
    max_acc = 0
    Acc = []
    print('Operating prediction based on the vectorized descriptors...')
    time.sleep(0.5)
    print(f'Number Of Entries:{len(X)}')
    dX = [(np.array(dscp) - np.array(dscp).mean(axis=0))[:, :50] for dscp in X]
    X = np.array([np.concatenate([np.array(x), dx], axis=1) for (x, dx) in zip(X, dX)])
    Y = np.array(Y)
    kf = KFold(n_splits=num_fold, shuffle=True)

    for kf_num, (train_index, test_index) in enumerate(tqdm(kf.split(X, Y))):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        X_train = [i for j in X_train for i in j]
        y_train = [i for j in y_train for i in j]
        rf = RandomForestRegressor(oob_score=True, n_estimators=200)
        model = rf
        acc = 0
        model.fit(X_train, y_train)
        for i, xtest in enumerate(X_test):
            y_pred = model.predict(xtest)
            if jobtype == 'regio':
                pred_value = np.array([p[0] - p[1] - p[2] + p[3] for p in y_pred.reshape(-1, 4)]) # enumerating all possible symmetry
                if np.argmax(pred_value) == np.array(
                        [y_test[i][j] for j in range(len(y_test[i])) if j % 4 == 0]).argmax():
                    acc += 1
            elif jobtype == 'all':
                pred_value1 = np.array([p[0] for p in y_pred.reshape(-1, 4)])
                pred_value2 = np.array([p[0] - p[1] - p[2] + p[3] for p in y_pred.reshape(-1, 4)])
                prob = (np.argsort(pred_value1)[-1], np.argsort(pred_value1)[-2])
                if pred_value2[prob[0]] > pred_value2[prob[1]]:
                    pred_product = prob[0]
                else:
                    pred_product = prob[1]
                if pred_product == np.array(
                        [y_test[i][j] for j in range(len(y_test[i])) if j % 4 == 0]).argmax():
                    acc += 1
            max_acc = acc / len(y_test)
        print(f'Curring Accuracy: {acc / len(y_test)}')
        joblib.dump(model, f'saved_models/{num_fold}_{model_name}')

        Acc.append(acc / (len(y_test)))
    print(f'There are prediction outputs after {num_fold}-times training based on RandomForest.')
    print('Acc:', '%.4f' % np.array(Acc).mean() + ' ± ' + '%.4f' % np.array(Acc).std())
    print(f'Best Model has been saved as {model_name} in saved_models folder.\n')


def predict_single_DA(sml, model_name='all.model', jobtype='', temp=20, acid=0):
    try:
        temp, acid = float(temp), float(acid)
        if jobtype == '':
            jobtype = 'all'
        r1_sml, r2_sml = sml.split('.')
        products_run, r1_sml, r2_sml = operate_DA(r1_sml, r2_sml)
        if len(products_run) == 1:
            print('This is the only possible reaction.')
            print(f'The product is: {products_run[0]}\n')
            print(f'The whole reaction SMILES is: {r1_sml}.{r2_sml}>>{products_run[0]}\n')
        else:
            dscp, _ = get_DA_dscp(f'{r1_sml}.{r2_sml}>>{products_run[0]}', products_run, jobtype, temp, acid)
            # print(dscp)
            ddscp = (np.array(dscp) - np.array(dscp).mean(axis=0))[:, :50]
            dscp = np.concatenate([np.array(dscp), ddscp], axis=1)
            model = joblib.load(f'saved_models/{model_name}')
            outcomes = np.array(model.predict(dscp))
            if jobtype in ['regio']:
                pred_value = np.array([p[0] - p[1] - p[2] + p[3] for p in outcomes.reshape(-1, 4)])
                print(f'The major product is: {products_run[np.argmax(pred_value)]}\n')
                print(f'The whole reaction SMILES is: {r1_sml}.{r2_sml}>>{products_run[np.argmax(pred_value)]}\n')
            elif jobtype == 'all':
                pred_value1 = np.array([p[0] for p in outcomes.reshape(-1, 4)])
                pred_value2 = np.array([p[0] - p[1] - p[2] + p[3] for p in outcomes.reshape(-1, 4)])
                prob = (np.argsort(pred_value1)[-1], np.argsort(pred_value1)[-2])
                if pred_value2[prob[0]] > pred_value2[prob[1]]:
                    pred_product = prob[0]
                else:
                    pred_product = prob[1]
                print(f'The major product is: {products_run[pred_product]}\n')
                print(f'The whole reaction SMILES is: {r1_sml}.{r2_sml}>>{products_run[pred_product]}\n')
                print(f'Here are minor products: ' + '.'.join(
                [product for product in products_run if product != products_run[pred_product]]) + '\n')
    except:
        print('There\'s something wrong in your input, please check again.')

def predict_file_DA(filepath='DA_19_all.csv', model_name='all.model', save_name='predict_output.csv', jobtype=''):
    if jobtype == '':
        jobtype = 'regio' if 'regio' in filepath else ('all' if 'all' in filepath else '')
    df = pd.read_csv(f'datasets/{filepath}')
    corr_sign = []
    predicts = []
    has_product_flag = True if '>>' in df.rxn_smiles.values[0] else False
    for i in trange(df.shape[0]):
        if has_product_flag:
            product = df['rxn_smiles'][i][df['rxn_smiles'][i].find('>>') + 2:]
            reactants = df['rxn_smiles'][i][:df['rxn_smiles'][i].find('>>')].split('.')
        else:
            reactants = df['rxn_smiles'][i].split('.')
        products_run = operate_DA(reactants[0], reactants[1])[0]
        dscp, _ = get_DA_dscp(df['rxn_smiles'][i], products_run, jobtype, df['Temp'][i], df['Lewis_acid'][i])
        ddscp = (np.array(dscp) - np.array(dscp).mean(axis=0))[:, :50]
        dscp = np.concatenate([np.array(dscp), ddscp], axis=1)
        model = joblib.load(f'saved_models/{model_name}')
        outcomes = np.array(model.predict(dscp))
        if jobtype == 'regio':
            pred_value = np.array([p[0] - p[1] - p[2] + p[3] for p in outcomes.reshape(-1, 4)])
            predicted_product = products_run[np.argmax(pred_value)]
        elif jobtype == 'all':
            pred_value1 = np.array([p[0] for p in outcomes.reshape(-1, 4)])
            pred_value2 = np.array([p[0] - p[1] - p[2] + p[3] for p in outcomes.reshape(-1, 4)])
            prob = (np.argsort(pred_value1)[-1], np.argsort(pred_value1)[-2])
            if pred_value2[prob[0]] > pred_value2[prob[1]]:
                pred_product = prob[0]
            else:
                pred_product = prob[1]
            predicted_product = products_run[pred_product]
        predicts.append(predicted_product)
        if has_product_flag:
            if Chem.MolToInchi(Chem.MolFromSmiles(product)) == Chem.MolToInchi((Chem.MolFromSmiles(predicted_product))):
                corr_sign.append(1)
            else:
                corr_sign.append(0)
    # print(corr_sign)
    if has_product_flag:
        df['corr_sign'] = corr_sign
        acc = sum(corr_sign) / df.shape[0]
        print('Prediction accuracy is :' + '%.4f' % acc)
    df['predicted'] = predicts
    df.to_csv(f'outputs/{save_name}', index=False)

    print(f'Reaction selectivity prediction has been saved as {save_name} in outputs folder.\n')



if args.mode == 'retrain' and args.dataset != '':
    classifier_single_DA(args.times_retrain, args.model_name, args.recalc, args.dataset, args.DA_jobtype)
elif args.mode == 'retrain' and args.dataset == '':
    classifier_single_DA(args.times_retrain, args.model_name, args.recalc, jobtype=args.DA_jobtype)
if args.mode == 'predict':
    predict_single_DA(args.rxn_sml, args.model_name, args.DA_jobtype, args.temp, args.acid)
if args.mode == 'predict_file':
    predict_file_DA(args.rxn_sml, args.model_name, args.save_name, args.DA_jobtype)
