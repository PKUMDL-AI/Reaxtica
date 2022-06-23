import os
import sys
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from qmdesc import ReactivityDescriptorHandler
from tqdm import trange, tqdm
from matplotlib import pyplot as plt
import argparse

sys.path.append('..')

from Utils.TSEI import calc_TSEI
from Utils.sterimol.sterimol import calc_steric


parser = argparse.ArgumentParser(description='Reaxtica: Predicting ddG for Chiral Phosphate Asymmetric Catalysis')
parser.add_argument('-m', '--mode', default='predict', choices=['predict', 'predict_file', 'retrain'],
                    help='Select whether to retrain the model or utilize existed model to predict.')
parser.add_argument('-rxn', '--rxn_sml', default='', help='The reaction/reaction file you intend to predict.')
parser.add_argument('--model_name', default='CPA_best.model', help='Set the name of your trained model.')
parser.add_argument('--save_name', default='predict_output.csv', help='Set the name of your predict output.')
parser.add_argument('--recalc', default=False, help='Whether to recalculate the descriptors.')
parser.add_argument('-dataset', '--dataset', default='', help='The dataset you are going to train or predict.')
args = parser.parse_args()

B_dict = {'O=P1(O)OC2=C(C3=C(F)C=C(OC)C=C3F)C=C4C(C=CC=C4)=[C@]2[C@]5=C(O1)C(C6=C(F)C=C(OC)C=C6F)=CC7=C5C=CC=C7': (
    8.90106605997137, 1.7191601056396644, 3.714652652864814),
    'O=P1(O)OC2=C(C3=CC(C)=C(OC(C)C)C(C)=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC(C)=C(OC(C)C)C(C)=C6)=CC7=C5C=CC=C7': (
        9.622080631068888, 2.024376918740776, 4.363723531507992),
    'O=P1(O)OC2=C(C3=CC=C(S(F)(F)(F)(F)F)C=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC=C(S(F)(F)(F)(F)F)C=C6)=CC7=C5C=CC=C7': (
        9.060550855504365, 2.582123208472027, 3.414323185992901),
    'O=P1(O)OC2=C(C3=CC(C4=CC(C(F)(F)F)=CC(C(F)(F)F)=C4)=CC(C5=CC(C(F)(F)F)=CC(C(F)(F)F)=C5)=C3)C=C6C(C=CC=C6)=C2C7=C(O1)C(C8=CC(C9=CC(C(F)(F)F)=CC(C(F)(F)F)=C9)=CC(C%10=CC(C(F)(F)F)=CC(C(F)(F)F)=C%10)=C8)=CC%11=C7C=CC=C%11': (
        9.778667921989204, 4.417364055540813, 8.529279057569795),
    'O=P1(O)OC2=C(CC3=CC=C(OC)C=C3)C=C4C(CCCC4)=C2C5=C(O1)C(CC6=CC=C(OC)C=C6)=CC7=C5CCCC7': (
        6.036639150387767, 1.90021848379649, 8.039512542130979),
    'O=P1(O)OC2=C(CC3=CC(C(F)(F)F)=CC(C(F)(F)F)=C3)C=C4C(CCCC4)=C2C5=C(O1)C(CC6=CC(C(F)(F)F)=CC(C(F)(F)F)=C6)=CC7=C5CCCC7': (
        6.08399881352612, 1.9001168896701528, 7.25213139969976),
    'O=P1(O)OC2=C(CC3=C(C=CC=C4)C4=CC5=C3C=CC=C5)C=C6C(C=CC=C6)=C2C7=C(O1)C(CC8=C(C=CC=C9)C9=CC%10=C8C=CC=C%10)=CC%11=C7C=CC=C%11': (
        5.3020598966353845, 1.9064046234600673, 7.084105366363846),
    'O=P1(O)OC2=C(C3=CC(C4=CC=C(OC)C=C4)=CC(C5=CC=C(OC)C=C5)=C3)C=C6C(C=CC=C6)=C2C7=C(O1)C(C8=CC(C9=CC=C(OC)C=C9)=CC(C%10=CC=C(OC)C=C%10)=C8)=CC%11=C7C=CC=C%11': (
        9.59188721091309, 2.6577819779945635, 8.975697567908464),
    'O=P1(O)OC2=C(Br)C=C3C(C=CC=C3)=C2C4=C(O1)C(Br)=CC5=CC=CC=C54': (
        4.2849488354274605, 1.9499986051062674, 2.037162444230757),
    'O=P1(O)OC2=C([Si](C3=CC=CC=C3)(C4=CC=CC=C4)C5=CC=CC=C5)C=C6C(CCCC6)=C2C7=C(O1)C([Si](C8=CC=CC=C8)(C9=CC=CC=C9)C%10=CC=CC=C%10)=CC%11=C7CCCC%11': (
        6.413374950789362, 4.330256653041993, 6.396989869936308),
    'O=P1(O)OC2=C([Si](C3=CC=C(C(C)(C)C)C=C3)(C4=CC=C(C(C)(C)C)C=C4)C5=CC=C(C(C)(C)C)C=C5)C=C6C(CCCC6)=C2C7=C(O1)C([Si](C8=CC=C(C(C)(C)C)C=C8)(C9=CC=C(C(C)(C)C)C=C9)C%10=CC=C(C(C)(C)C)C=C%10)=CC%11=C7CCCC%11': (
        7.730537450502509, 5.840247391547069, 8.89133832594083),
    'O=P1(O)OC2=C(C3=CC=C(C4=CC(C(F)(F)F)=CC(C(F)(F)F)=C4)C=C3)C=C5C(C=CC=C5)=C2C6=C(O1)C(C7=CC=C(C8=CC(C(F)(F)F)=CC(C(F)(F)F)=C8)C=C7)=CC9=C6C=CC=C9': (
        11.700230393712241, 2.6531523244570434, 5.045819017952461),
    'O=P1(O)OC2=C(C3=CC(C4=C(C)C=C(C)C=C4C)=CC(C5=C(C)C=C(C)C=C5C)=C3)C=C6C(CCCC6)=C2C7=C(O1)C(C8=CC(C9=C(C)C=C(C)C=C9C)=CC(C%10=C(C)C=C(C)C=C%10C)=C8)=CC%11=C7CCCC%11': (
        8.921748489905939, 4.287988663477968, 8.015894433516994),
    'O=P1(O)OC2=C(C3=C(C(C)C)C=C(C4=CC=C(C(C)(C)C)C=C4)C=C3C(C)C)C=C5C(C=CC=C5)=[C@]2[C@]6=C(O1)C(C7=C(C(C)C)C=C(C8=CC=C(C(C)(C)C)C=C8)C=C7C(C)C)=CC9=C6C=CC=C9': (
        13.381166983275984, 3.059672658238724, 5.612707861800768),
    'O=P1(O)OC2=C(CC)C=C3C(CCCC3)=C2C4=C(O1)C(CC)=CC5=C4CCCC5': (
        4.574198697535736, 1.8848843970906008, 3.1550657713406607),
    'O=P1(O)OC2=C(C3=C(C=CC=C4)C4=C(C5=CC(C=CC=C6)=C6C=C5)C7=C3C=CC=C7)C=C8C(C=CC=C8)=[C@]2[C@]9=C(O1)C(C%10=C(C=CC=C%11)C%11=C(C%12=CC=C(C=CC=C%13)C%13=C%12)C%14=C%10C=CC=C%14)=CC%15=C9C=CC=C%15': (
        13.276833371073607, 3.153364767359537, 5.654917278259949),
    'O=P1(O)OC2=[C@]([C@]3=C(Cl)C=C(Cl)C=C3Cl)C=C4C(CCCC4)=[C@]2[C@]5=C(O1)C(C6=C(Cl)C=C(Cl)C=C6Cl)=CC7=C5CCCC7': (
        8.244222063599633, 1.7300696952846288, 4.548838843483974),
    'O=P1(O)OC2=C(C3=C(OCC)C=CC(C)=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC(C)=CC=C6OCC)=CC7=C5C=CC=C7': (
        6.788859259837708, 1.8320197725006389, 6.688729418118125),
    'O=P1(O)OC2=C(C3=CC(COC)=CC=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC=CC(COC)=C6)=CC7=C5C=CC=C7': (
        7.364753718783399, 2.0053612592378607, 5.36752428289628),
    'O=P1(O)OC2=C(C3=C(C4=CC(C=CC=C5)=C5C=C4)C=CC=C3)C=C6C(CCCC6)=C2C7=C(O1)C(C8=CC=CC=C8C9=CC=C(C=CC=C%10)C%10=C9)=CC%11=C7CCCC%11': (
        6.799678343795056, 2.1096748476020513, 8.969659137516803),
    'O=P1(O)OC2=C(C3=CC(C4=CC(C=CC=C5)=C5C=C4)=CC=C3)C=C6C(C=CC=C6)=C2C7=C(O1)C(C8=CC=CC(C9=CC=C(C=CC=C%10)C%10=C9)=C8)=CC%11=C7C=CC=C%11': (
        9.222237899681458, 1.975674056107612, 8.773517842329214),
    'O=P1(O)OC2=C(C3=CC=C(C4CCCCC4)C=C3)C=C5C(CCCC5)=C2C6=C(O1)C(C7=CC=C(C8CCCCC8)C=C7)=CC9=C6CCCC9': (
        11.018659692396765, 2.467054882435962, 3.5643291182237795),
    'O=P1(O)OC2=C(C3=CC(C(F)(F)F)=CC(C(F)(F)F)=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC(C(F)(F)F)=CC(C(F)(F)F)=C6)=CC7=C5C=CC=C7': (
        7.342649457661363, 2.0532384202261946, 4.945744093190028),
    'O=P1(O)OC2=C(C3=CC(C(F)(F)F)=CC(C(F)(F)F)=C3)C=C4C(CCCC4)=C2C5=C(O1)C(C6=CC(C(F)(F)F)=CC(C(F)(F)F)=C6)=CC7=C5CCCC7': (
        7.339101481773609, 2.1335954439661626, 4.951071032987724),
    'O=P1(O)OC2=C(C3=CC=CC=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC=CC=C6)=CC7=C5C=CC=C7': (
        6.788624040853299, 1.700000588919937, 3.15630997226753),
    'O=P1(O)OC2=C(C3=C(C)C=C(C)C=C3C)C=C4C(C=CC=C4)=C2C5=C(O1)[C@@]([C@@]6=C(C)C=C(C)C=C6C)=CC7=C5C=CC=C7': (
        7.738235955900351, 1.921813848920409, 4.380999161979537),
    'O=P1(O)OC2=C(C3=C(C(C)C)C=C(C(C)C)C=C3C(C)C)C=C4C(C=CC=C4)=C2C5=C(O1)[C@@]([C@@]6=C(C(C)C)C=C(C(C)C)C=C6C(C)C)=CC7=C5C=CC=C7': (
        9.017120118579804, 3.066617470533308, 5.610511911477974),
    'O=P1(O)OC2=C(C3=C(C4CCCCC4)C=C(C5CCCCC5)C=C3C6CCCCC6)C=C7C(C=CC=C7)=C2C8=C(O1)[C@@]([C@@]9=C(C%10CCCCC%10)C=C(C%11CCCCC%11)C=C9C%12CCCCC%12)=CC%13=C8C=CC=C%13': (
        10.556445127288418, 2.937650670184271, 7.429689886257721),
    'O=P1(O)OC2=C(CC3=CC=C(C(F)(F)F)C=C3C(F)(F)F)C=C4C(C=CC=C4)=C2C5=C(O1)C(CC6=C(C(F)(F)F)C=C(C(F)(F)F)C=C6)=CC7=C5C=CC=C7': (
        6.488786782604458, 1.895507185576398, 7.697324716206317),
    'O=P1(O)OC2=C([Si](C3=CC=CC=C3)(C)C4=CC=CC=C4)C=C5C(C=CC=C5)=C2C6=C(O1)C([Si](C7=CC=CC=C7)(C8=CC=CC=C8)C)=CC9=C6C=CC=C9': (
        6.954719282416205, 3.157120051811307, 6.486042157925494),
    'O=P1(O)OC2=C(C3=CC(C(C)(C)C)=CC(C(C)(C)C)=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC(C(C)(C)C)=CC(C(C)(C)C)=C6)=CC7=C5C=CC=C7': (
        7.789515975740965, 3.085087394012793, 5.728250533811211),
    'O=P1(O)OC2=C(C3=CC(C(C)(C)C)=CC(C(C)(C)C)=C3)C=C4C(CCCC4)=C2C5=C(O1)C(C6=CC(C(C)(C)C)=CC(C(C)(C)C)=C6)=CC7=C5CCCC7': (
        7.472040900518496, 3.081011001883112, 5.732591679445384),
    'O=P1(O)OC2=C(Br)C=C3C(CCCC3)=C2C4=C(O1)C(Br)=CC5=C4CCCC5': (
        4.28491698143061, 1.9499985888608553, 2.037152253718799),
    'O=P1(O)OC2=C([Si](C3=CC=CC=C3)(C4=CC=CC=C4)C5=CC=CC=C5)C=C6C(C=CC=C6)=C2C7=C(O1)C([Si](C8=CC=CC=C8)(C9=CC=CC=C9)C%10=CC=CC=C%10)=CC%11=C7C=CC=C%11': (
        6.41310724818389, 4.330395025546664, 6.396964826183483),
    'O=P1(O)OC2=C(C3=CC=C(OC)C=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC=C(OC)C=C6)=CC7=C5C=CC=C7': (
        8.89846225375593, 1.722991497164872, 3.1692441641033366),
    'O=P1(O)OC2=C(C3=CC(COC)=CC=C3)C=C4C(CCCC4)=C2C5=C(O1)C(C6=CC=CC(COC)=C6)=CC7=C5CCCC7': (
        6.879128474863526, 1.7458301621774677, 5.899725300607959),
    'O=P1(O)OC2=C(C3=CC=C(C)C=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC=C(C)C=C6)=CC7=C5C=CC=C7': (
        7.714468866037532, 1.7327470923367174, 3.1565393776438224),
    'O=P1(O)OC2=C(C3=C(OC(F)(F)F)C=CC=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC=CC=C6OC(F)(F)F)=CC7=C5C=CC=C7': (
        6.789045010906426, 1.88494775425828, 5.9288428740068255),
    'O=P1(O)OC2=C(C3=C(C=CC=C4)C4=CC5=C3C=CC=C5)C=C6C(C=CC=C6)=C2C7=C(O1)[C@@]([C@@]8=C(C=CC=C9)C9=CC%10=C8C=CC=C%10)=CC%11=C7C=CC=C%11': (
        6.857549622255069, 1.7491871292957575, 5.665509502885394),
    'O=P1(O)OC2=C(C3=CC=C(C(C)(C)C)C=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC=C(C(C)(C)C)C=C6)=CC7=C5C=CC=C7': (
        9.031290885208083, 2.639994834181725, 3.3024386659549476),
    'O=P1(O)OC2=C(C3=C(C=CC4=CC=CC(C=C5)=C46)C6=C5C=C3)C=C7C(C=CC=C7)=C2C8=C(O1)C(C9=CC=C(C=C%10)C%11=C9C=CC%12=CC=CC%10=C%11%12)=CC%13=C8C=CC=C%13': (
        9.110633876203194, 1.8369916279488037, 6.733132215885712),
    'O=P1(O)OC2=C(C3=CC=C(C4=CC=C(C=CC=C5)C5=C4)C=C3)C=C6C(C=CC=C6)=C2C7=C(O1)C(C8=CC=C(C9=CC(C=CC=C%10)=C%10C=C9)C=C8)=CC%11=C7C=CC=C%11': (
        13.241507970604292, 2.3160507249221296, 4.312332633189627),
    'O=P1(O)OC2=C(C3=C(OC)C=CC=C3OC)C=C4C(C=CC=C4)=C2C5=C(O1)[C@@]([C@@]6=C(OC)C=CC=C6OC)=CC7=C5C=CC=C7': (
        6.795751603136252, 1.8100690190677646, 5.399435107136813)}


def Boltzmann_normalize(conf_energy, properties):
    conf_energy = conf_energy - conf_energy.min()
    return np.sum([np.exp(-e * 1000 * 4.18 / 298.15 * 8.3144) * property for (e, property) in
                   zip(conf_energy, properties)]) / np.sum(
        [np.exp(-e * 1000 * 4.18 / 298.15 * 8.3144) for e in conf_energy])


catalyst_num = 0


def get_qmdesc_CPA(x):
    global B_dict, catalyst_num
    try:
        catalyst, imine, thiol = Chem.MolFromSmiles(x['Catalyst']), Chem.MolFromSmiles(
            x['Imine']), Chem.MolFromSmiles(x['Thiol'])
        c_sml, i_sml, t_sml = x['Catalyst'], x['Imine'], x['Thiol']
    except:
        catalyst, imine, thiol = Chem.MolFromSmiles(x.split('.')[0]), Chem.MolFromSmiles(
            x.split('.')[1]), Chem.MolFromSmiles(x.split('.')[2])
        c_sml, i_sml, t_sml = x.split('.')[0], x.split('.')[1], x.split('.')[2]

    infos = []
    for idx, sml in enumerate([c_sml, i_sml, t_sml]):
        handler = ReactivityDescriptorHandler()
        infos.append(handler.predict(sml))
    tseis = []
    for idx, mol in enumerate([catalyst, imine, thiol]):
        tseis.append(
            [calc_TSEI(mol, [i for i in range(len(mol.GetAtoms())) if i != j], j) for j in range(len(mol.GetAtoms()))])
    descriptor = []

    catalyst_aromatic_nums = []
    for atom in catalyst.GetAtoms():
        if atom.GetAtomicNum() == 15:
            catalyst_P_num = atom.GetIdx()
        if atom.GetAtomicNum() == 6 and atom.GetIsAromatic():
            catalyst_aromatic_nums.append(atom.GetIdx())
    descriptor += ([infos[0][j][catalyst_P_num] for j in list(infos[0].keys())[1:4]] + [
        infos[0]['NMR'][catalyst_P_num]])
    additive_O1_num, additive_O2_num, additive_C_num, PO_bond_nums = [-1, -1], [], [], [-1, -1]
    for atom in catalyst.GetAtoms()[catalyst_P_num].GetNeighbors():
        if atom.GetAtomicNum() == 8 and (6 not in [a.GetAtomicNum() for a in atom.GetNeighbors()]):
            if catalyst.GetBondBetweenAtoms(atom.GetIdx(), catalyst_P_num).GetBondType() == 2:
                additive_O1_num[0] = atom.GetIdx()
                PO_bond_nums[0] = catalyst.GetBondBetweenAtoms(atom.GetIdx(), catalyst_P_num).GetIdx()
            else:
                additive_O1_num[1] = atom.GetIdx()
                PO_bond_nums[1] = catalyst.GetBondBetweenAtoms(atom.GetIdx(), catalyst_P_num).GetIdx()
        elif atom.GetAtomicNum() == 8 and (6 in [a.GetAtomicNum() for a in atom.GetNeighbors()]):
            additive_O2_num.append(atom.GetIdx())
    for atom in catalyst.GetAtoms()[additive_O2_num[0]].GetNeighbors() + catalyst.GetAtoms()[
        additive_O2_num[1]].GetNeighbors():
        if atom.GetAtomicNum() == 6:
            additive_C_num.append(atom.GetIdx())
    O1_descriptor, O2_descriptor, C_descriptor, arr_descriptor = [], [], [], []
    for O_num in additive_O1_num:
        O1_descriptor.append(
            [infos[0][j][O_num] for j in list(infos[0].keys())[1:4]] + [infos[0]['NMR'][O_num]])
    descriptor += np.array(O1_descriptor).mean(axis=0).tolist()
    for O_num in additive_O2_num:
        O2_descriptor.append(
            [infos[0][j][O_num] for j in list(infos[0].keys())[1:4]] + [infos[0]['NMR'][O_num]])
    descriptor += np.array(O2_descriptor).mean(axis=0).tolist()
    for C_num in additive_C_num:
        C_descriptor.append(
            [infos[0][j][C_num] for j in list(infos[0].keys())[1:4]] + [infos[0]['NMR'][C_num]])
    descriptor += np.array(C_descriptor).mean(axis=0).tolist()
    for bond_num in PO_bond_nums:
        descriptor += [infos[0][j][bond_num] for j in list(infos[0].keys())[5:7]]

    for atom in imine.GetAtoms():
        if atom.GetAtomicNum() == 7:
            base_N_num = atom.GetIdx()
            break
    descriptor += [infos[1][j][base_N_num] for j in list(infos[1].keys())[1:4]] + [infos[1]['NMR'][base_N_num]] + [
        tseis[1][base_N_num]]
    for atom in imine.GetAtoms()[base_N_num].GetNeighbors():
        if atom.GetAtomicNum() == 6 and (8 not in [a.GetAtomicNum() for a in atom.GetNeighbors()]):
            base_C_num = atom.GetIdx()
            break
    descriptor += [infos[1][j][base_C_num] for j in list(infos[1].keys())[1:4]] + [infos[1]['NMR'][base_C_num]] + [
        tseis[1][base_C_num]]
    base_bond_num = imine.GetBondBetweenAtoms(base_C_num, base_N_num).GetIdx()
    descriptor += [infos[1][j][base_bond_num] for j in list(infos[1].keys())[5:7]]

    for atom in thiol.GetAtoms():
        if atom.GetAtomicNum() == 16:
            thiol_S_num = atom.GetIdx()
            break
    descriptor += [infos[2][j][thiol_S_num] for j in list(infos[2].keys())[1:4]] + [infos[2]['NMR'][thiol_S_num]] + [
        tseis[2][thiol_S_num]]

    get_substituent = AllChem.ReactionFromSmarts(
        '[#8:1][P:2]1(=[O:3])[#8:17]-[c:16]2[c:18][c:19][c:20]3[c:25][c:24][c:23][c:22][c:21]3[c:15]2-[c:14]2[c:5](-[#8:4]1)[c:6][c:7][c:8]1[c:9][c:10][c:11][c:12][c:13]21>>[#6:18].[#6:6]')
    get_substituent2 = AllChem.ReactionFromSmarts(
        '[#8:1][P:2]1(=[O:3])[#8:17]-[c:16]2[c:18][c:19][c:20]3-[#6:25]-[#6:24]-[#6:23]-[#6:22]-[c:21]3[c:15]2-[c:14]2[c:13]3-[#6:12]-[#6:11]-[#6:10]-[#6:9]-[c:8]3[c:7][c:6][c:5]2-[#8:4]1>>[#6:18].[#6:6]')
    for idx, atom in enumerate(catalyst.GetAtoms()):
        atom.SetAtomMapNum(idx + 1)

    substituent = ([Chem.MolToSmiles(i[0]) for i in get_substituent.RunReactants((catalyst,))] + [Chem.MolToSmiles(i[0])
                                                                                                  for i in
                                                                                                  get_substituent2.RunReactants(
                                                                                                      (catalyst,))])[0]
    cat_sub_mol = Chem.MolFromSmiles(substituent)
    for atom in cat_sub_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            substitute_site = atom.GetIdx()
            substitute_nei_site = atom.GetNeighbors()[0].GetIdx()
    if c_sml in B_dict:
        descriptor += [B_dict[c_sml][0], B_dict[c_sml][1], B_dict[c_sml][2]]
    else:
        mol = Chem.AddHs(cat_sub_mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=5)
        opt_output = AllChem.MMFFOptimizeMoleculeConfs(mol)
        MMFF_energy = np.array([o[1] for o in opt_output])
        steric = []
        for conf_idx in range(5):
            Chem.MolToMolFile(mol, f'catalysts/catalyst{catalyst_num}_{conf_idx}.mol', confId=conf_idx)
            conv = openbabel.OBConversion()
            conv.SetInAndOutFormats('mol', 'gjf')
            convmol = openbabel.OBMol()
            conv.ReadFile(convmol, f'catalysts/catalyst{catalyst_num}_{conf_idx}.mol')
            conv.WriteFile(convmol, f'catalysts/catalyst{catalyst_num}_{conf_idx}.gjf')
            with open(f'catalysts/catalyst{catalyst_num}_{conf_idx}.gjf') as f:
                content = f.readlines()
            content = ['# test\n', '\n', 'title\n', '\n', '1 0\n'] + content[6:]
            with open(f'catalysts/catalyst{catalyst_num}_{conf_idx}.gjf', 'w') as f:
                f.writelines(content)
            steric.append(list(calc_steric(f'catalysts/catalyst{catalyst_num}_{conf_idx}.gjf', substitute_site + 1,
                                           substitute_nei_site + 1)))
        steric = np.array(steric)
        L, B1, B5 = Boltzmann_normalize(MMFF_energy, steric[:, 0]), Boltzmann_normalize(MMFF_energy, steric[:,
                                                                                                     1]), Boltzmann_normalize(
            MMFF_energy, steric[:, 2])
        B_dict[c_sml] = (L, B1, B5)
        descriptor += [B_dict[c_sml][2]]
        catalyst_num += 1

    return np.array(descriptor)


def gen_CPA_input(sheetname):
    df = pd.read_excel('datasets/CPA_input.xlsx', sheetname)
    dscp = df[['Catalyst', 'Imine', 'Thiol']].apply(lambda x: get_qmdesc_CPA(x), axis=1)
    dscp = np.array(dscp)
    np.save(f'saved_descriptors/CPA_{sheetname}_dscp.npy', dscp)


def retrain_yield_RF(sheetname, num_fold=5, model_name='best.model', recalc=False):
    analyzed = pd.read_excel('../Utils/analyzed_descriptor.xlsx').CPA_MAE_sort.values
    if sheetname == 'random':
        average_R2, average_RMSE, average_MAE = [], [], []
        split_num = 600
        for num in range(1, 10):
            sheet = f'FullCV_0{num}'
            if not os.path.exists(f'saved_descriptors/CPA_{sheet}_dscp.npy') or recalc:
                gen_CPA_input(sheet)
            X = np.load(f'saved_descriptors/CPA_{sheet}_dscp.npy', allow_pickle=True)
            Y = pd.read_excel('datasets/CPA_input.xlsx', sheet).Output.values
            X = np.array([x.reshape(-1) for x in X])
            # scaler = StandardScaler()
            # X = scaler.fit_transform(X)
            orig_X = X
            R2, RMSE, MAE = [], [], []
            max_R2 = 0
            X = orig_X[:, np.argsort(np.array(analyzed))[:24]]
            for _ in trange(num_fold):
                X_train, X_test, y_train, y_test = X[:split_num], X[split_num:], Y[:split_num], Y[split_num:]
                model = RandomForestRegressor(oob_score=True, n_estimators=500)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_train = model.predict(X_train)
                R2.append(r2_score(y_test, y_pred))
                RMSE.append(mean_squared_error(y_test, y_pred) ** 0.5)
                MAE.append(mean_absolute_error(y_test, y_pred))
                if max_R2 < r2_score(y_test, y_pred):
                    joblib.dump(model, f'saved_models/{sheet}_{model_name}')
                    max_R2 = r2_score(y_test, y_pred)
            print(f'Random Validation for CPA: {num}')
            print('RandomForest' + '|  R2_score:', '%.4f' % np.array(R2).mean() + ' ± ' + '%.4f' % np.array(R2).std(),
                  ' RMSE:', '%.3f' % np.array(RMSE).mean() + ' ± ' + '%.3f' % np.array(RMSE).std(),
                  ' MAE:', '%.3f' % np.array(MAE).mean() + ' ± ' + '%.3f' % np.array(MAE).std())
            average_R2.append(np.array(R2).mean())
            average_RMSE.append(np.array(RMSE).mean())
            average_MAE.append(np.array(MAE).mean())
        sheet = 'FullCV_10'
        if not os.path.exists(f'saved_descriptors/CPA_{sheet}_dscp.npy') or recalc:
            gen_CPA_input(sheet)
        X = np.load(f'saved_descriptors/CPA_{sheet}_dscp.npy', allow_pickle=True)
        Y = pd.read_excel('datasets/CPA_input.xlsx', sheet).Output.values
        X = np.array([x.reshape(-1) for x in X])
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)
        orig_X = X
        R2, RMSE, MAE = [], [], []
        max_R2 = 0
        X = orig_X[:, np.argsort(np.array(analyzed))[:24]]
        for _ in trange(num_fold):
            X_train, X_test, y_train, y_test = X[:split_num], X[split_num:], Y[:split_num], Y[split_num:]
            model = RandomForestRegressor(oob_score=True, n_estimators=500)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            R2.append(r2_score(y_test, y_pred))
            RMSE.append(mean_squared_error(y_test, y_pred) ** 0.5)
            MAE.append(mean_absolute_error(y_test, y_pred))
            if max_R2 < r2_score(y_test, y_pred):
                joblib.dump(model, f'saved_models/{sheet}_{model_name}')
                max_R2 = r2_score(y_test, y_pred)
        print(f'Random Validation for CPA: 10')
        print('RandomForest' + '|  R2_score:', '%.4f' % np.array(R2).mean() + ' ± ' + '%.4f' % np.array(R2).std(),
              ' RMSE:', '%.3f' % np.array(RMSE).mean() + ' ± ' + '%.3f' % np.array(RMSE).std(),
              ' MAE:', '%.3f' % np.array(MAE).mean() + ' ± ' + '%.3f' % np.array(MAE).std())
        average_R2.append(np.array(R2).mean())
        average_RMSE.append(np.array(RMSE).mean())
        average_MAE.append(np.array(MAE).mean())
        print('Average on 10 random validation for CPA')
        print('RandomForest' + '|  R2_score:', '%.4f' % np.array(average_R2).mean() + ' ± ' + '%.4f' % np.array(average_R2).std(),
              ' RMSE:', '%.3f' % np.array(average_RMSE).mean() + ' ± ' + '%.3f' % np.array(average_RMSE).std(),
              ' MAE:', '%.3f' % np.array(average_MAE).mean() + ' ± ' + '%.3f' % np.array(average_MAE).std())
    else:
        if 'FullCV' in sheetname:
            split_num = 600
        else:
            split_num = 384
        if not os.path.exists(f'saved_descriptors/CPA_{sheetname}_dscp.npy') or recalc:
            gen_CPA_input(sheetname)
        X = np.load(f'saved_descriptors/CPA_{sheetname}_dscp.npy', allow_pickle=True)
        Y = pd.read_excel('datasets/CPA_input.xlsx', sheetname).Output.values
        X = np.array([x.reshape(-1) for x in X])
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)
        orig_X = X
        R2, RMSE, MAE = [], [], []
        max_R2 = 0
        X = orig_X[:, np.argsort(np.array(analyzed))[:24]]
        for _ in trange(num_fold):
            X_train, X_test, y_train, y_test = X[:split_num], X[split_num:], Y[:split_num], Y[split_num:]
            model = RandomForestRegressor(oob_score=True, n_estimators=500)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            R2.append(r2_score(y_test, y_pred))
            RMSE.append(mean_squared_error(y_test, y_pred) ** 0.5)
            MAE.append(mean_absolute_error(y_test, y_pred))
            if max_R2 < r2_score(y_test, y_pred):
                joblib.dump(model, f'saved_models/{sheetname}_{model_name}')
                max_R2 = r2_score(y_test, y_pred)
                '''
                if _ == 4:
                plt.scatter(y_train, y_pred_train, c='r', alpha=0.2, label='Train')
                plt.scatter(y_test, y_pred, alpha=0.4, label='Predicted')
                plt.xlim(-1, 5)
                plt.ylim(-1, 5)
                plt.plot([-1, 5], [-1, 5])
                plt.xlabel('Observed ddG')
                plt.ylabel('Predicted ddG')
                plt.title(f'R2: {round(R2[-1], 4)}\nRMSE: {round(RMSE[-1], 2)}\nMAE: {round(MAE[-1], 2)}')
                plt.legend()
                plt.show()'''
        print(f'There are prediction outputs after {num_fold}-times training based on RandomForest:')
        print('RandomForest' + '|  R2_score:', '%.4f' % np.array(R2).mean() + ' ± ' + '%.4f' % np.array(R2).std(),
              ' RMSE:', '%.3f' % np.array(RMSE).mean() + ' ± ' + '%.3f' % np.array(RMSE).std(),
              ' MAE:', '%.3f' % np.array(MAE).mean() + ' ± ' + '%.3f' % np.array(MAE).std())


def predict_single_CPA(x, model_name='CPA_best.model'):
    try:
        analyzed = pd.read_excel('../Utils/analyzed_descriptor.xlsx').CPA_MAE_sort.values
        dscp = get_qmdesc_CPA(x).reshape(1, -1)[:, np.argsort(np.array(analyzed))[:24]]
        model = joblib.load(f'saved_models/{model_name}')
        deltaG_pred = model.predict(dscp.reshape(1, -1))
        print(f'The ddG of this reaction is predicted to be {round(deltaG_pred.item(), 3)} kcal/mol.')
    except:
        print('There\'s something wrong in your input, please follow the $catalyst.$imine.$thiol format.')


def predict_file_CPA(filepath, model_name='CPA_best.model', save_name='predict_output.csv'):
    analyzed = pd.read_excel('../Utils/analyzed_descriptor.xlsx').CPA_MAE_sort.values
    df = pd.read_csv(f'datasets/{filepath}')
    dscp = df[['Catalyst', 'Imine', 'Thiol']].apply(lambda x: get_qmdesc_CPA(x), axis=1)
    dscp = np.array([x.reshape(-1) for x in dscp])
    dscp = dscp[:, np.argsort(np.array(analyzed))[:24]]
    model = joblib.load(f'saved_models/{model_name}')
    yield_pred = model.predict(dscp)
    df['yield'] = yield_pred
    df.to_csv(f'datasets/{save_name}', index=None)
    print(f'Reaction yield prediction has been saved as {save_name} in corresponding folder.\n')


if args.mode == 'predict':
    predict_single_CPA(args.rxn_sml, args.model_name)
elif args.mode == 'retrain':
    retrain_yield_RF(args.dataset, model_name=args.model_name, recalc=args.recalc)
elif args.mode == 'predict_file':
    predict_file_CPA(args.dataset, args.model_name, args.save_name)
