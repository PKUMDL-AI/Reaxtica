import threading
import requests
import re
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from threading import Thread
import time
import sys

sys.path.append('..')

from Utils.mol_utils import retro_DA_SMARTS

lock = threading.Lock()

def Grzybowski_predict(sml_input):
    # print(sml_input)
    diene, dienophile = sml_input.split('.')[0], sml_input.split('.')[1]
    headers = {'Host': 'dielsalderapp.grzybowskigroup.pl', 'Connection': 'keep-alive',
               'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Mobile Safari/537.36',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
               'Accept-Encoding': 'gzip, deflate',
               'Accept-Language': 'zh-CN,zh;q=0.9'}
    session = requests.session()
    session.get('http://dielsalderapp.grzybowskigroup.pl/')
    token = session.cookies['csrftoken']
    response = session.post('http://dielsalderapp.grzybowskigroup.pl/predict/',
                            data={'diene': diene, 'dienophile': dienophile,
                                  'csrfmiddlewaretoken': token},
                            timeout=30, headers=headers)
    # print(response.text)
    G_product = re.search(r'Major Product\</h2\>SMILES: .*\<br\>', response.text).group()[26:-4]
    return G_product



df = pd.read_csv('datasets/DA_19_all.csv')
Grzybowski_predicted = []
corr_sign = []
error_smiles = []
def operate_predict(rxn):
    with pool_sema:
        lock.acquire()
        order = retro_DA_SMARTS(rxn)
        r1, r2 = rxn[:rxn.find('>>')].split('.')
        r1_mol, r2_mol = Chem.MolFromSmiles(r1), Chem.MolFromSmiles(r2)
        for atom in r1_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        for atom in r2_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        r1, r2 = Chem.MolToSmiles(r1_mol), Chem.MolToSmiles(r2_mol)
        product = rxn[rxn.find('>>') + 2:]
        p_mol = Chem.MolFromSmiles(product)
        try:
            if order == 1:
                reactants = f'{r1}.{r2}'
                # print(reactants)
                predict_product = Grzybowski_predict(reactants)
            elif order == 0:
                reactants = f'{r2}.{r1}'
                # print(reactants)
                predict_product = Grzybowski_predict(reactants)
            else:
                raise ValueError
            Grzybowski_predicted.append(predict_product)
            if p_mol.HasSubstructMatch(Chem.MolFromSmiles(predict_product)) or Chem.MolToInchi(p_mol) == Chem.MolToInchi(Chem.MolFromSmiles(predict_product)):
                corr_sign.append(1) # the true product is predicted
            else:
                corr_sign.append(0) # the false product
        except: # when the websever gives no product
            Grzybowski_predicted.append('')
            corr_sign.append(0)
            error_smiles.append('\n'.join([rxn, r1, r2]))
        lock.release()

pool_sema = threading.BoundedSemaphore(20)
ts = [Thread(target=operate_predict, args=(rxn_sml, )) for rxn_sml in df.rxn_smiles]
for t in ts:
    t.start()
for t in ts:
    t.join()
print('%.4f' % (sum(corr_sign) / df.shape[0]))
df['corr_sign'] = corr_sign
df['Grzybowski_predicted'] = Grzybowski_predicted
df.to_csv('outputs/DA_Grzybowski_predicted_all_new.csv', index=False)

