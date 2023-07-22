from rdkit import Chem
import numpy as np
import pandas as pd
import networkx as nx

# Special thanks to Yingze Wang in CCME, Peking Univ. for contributing this script.

# From CRC Handbook 97ed

cov_rads = {"H": 0.32, "C": 0.75, "N": 0.71, "O": 0.64,
            "P": 1.09, "S": 1.04, "Si": 1.14, "F": 0.60,
            "Cl": 1.00, "Br": 1.17, "I": 1.36, "*": 0.75,
            'Mg': 1.40, 'Se': 1.18, 'Sn': 1.40, 'B': 0.82,
            'Na': 1.50, 'K': 2.27, 'Cs': 2.67, 'Ca': 1.97, 'Li': 1.52, 'Pd': 1.37}
lcc = cov_rads["C"] * 2


def convert_mol_to_graph(mol):
    graph = nx.Graph()
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), weight=1)
    return graph


def get_bond_length(conf, atomId1, atomId2):
    if isinstance(conf, Chem.Conformer):
        pos1 = np.array(conf.GetAtomPosition(atomId1))
        pos2 = np.array(conf.GetAtomPosition(atomId2))
        return np.sqrt(((pos1 - pos2) ** 2).sum())
    elif isinstance(conf, Chem.Mol):
        return cov_rads[conf.GetAtomWithIdx(atomId1).GetSymbol()] + cov_rads[conf.GetAtomWithIdx(atomId2).GetSymbol()]


def get_total_bond_length(conf, path):
    total_bond_length = 0
    for i in range(len(path) - 1):
        total_bond_length += get_bond_length(conf, path[i], path[i + 1])
    return total_bond_length


def calc_TSEI(mol, subs, centerId, includeHs=True):
    global cov_rads, lcc
    graph = convert_mol_to_graph(mol)
    TSEI = 0
    for atomId in subs:
        if (not includeHs) and (mol.GetAtomWithIdx(atomId).GetSymbol() == "H"):
            pass
        else:
            try:
                path = nx.shortest_path(graph, centerId, atomId)
                L_rel = get_total_bond_length(mol, path) / lcc
                R_rel = cov_rads[mol.GetAtomWithIdx(atomId).GetSymbol()] / cov_rads["C"]
                TSEI += R_rel ** 3 / L_rel ** 3
            except:
                pass

    return TSEI
'''testmol = Chem.MolFromSmiles('[N+:10]([c:11]1[cH:12][cH:13][c:14]([C:15]([Cl:16])=[O:17])[cH:18][cH:19]1)([O-:20])=[O:21]')
print([calc_TSEI(testmol, [i for i in range(len(testmol.GetAtoms())) if i != j], j) for j in range(len(testmol.GetAtoms()))])
testmol = Chem.AddHs(testmol)
print([calc_TSEI(testmol, [i for i in range(len(testmol.GetAtoms())) if i != j], j) for j in range(len(testmol.GetAtoms()))])'''

def main():
    from matplotlib import pyplot as plt
    df = pd.read_csv('TSEI_sample.csv', index_col=None)
    this_TSEI = []
    for i in range(df.shape[0]):
        test_mol = Chem.MolFromSmiles(df.iloc[i]['SMILES'])
        test_mol = Chem.AddHs(test_mol)
        # print(Chem.MolToSmiles(test_mol))
        # print(len(test_mol.GetAtoms()))
        this_TSEI.append(calc_TSEI(test_mol, [x for x in range(1, len(test_mol.GetAtoms()))], 0, includeHs=True))
    plt.scatter(df['TSEI'], this_TSEI)
    stdline = np.arange(0, 2.5, 0.01)
    plt.xlim(0, 2.5)
    plt.ylim(0, 2.5)
    plt.plot(stdline, stdline, color='black')
    plt.show()
