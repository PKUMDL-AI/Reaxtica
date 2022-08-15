import os
import sys
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from tqdm import trange, tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from rdkit.Chem import AllChem
from matplotlib import pyplot as plt
sys.path.append('..')

Reaxtica_X = np.load('saved_descriptors/SMC_dscp_USPTO.npy', allow_pickle=True)
Reaxtica_X = np.array([np.array(x) for x in Reaxtica_X])
drfp_X = np.load('saved_descriptors/fps_drfp_USPTO.npy', allow_pickle=True)
rxnfp_X = np.load('saved_descriptors/fps_rxnfp_USPTO.npy', allow_pickle=True)

Y = np.array(pd.read_csv('datasets/USPTO_SMC.csv').rxn_yield.values)

def training(X, Y, method_name):
    R2, RMSE, MAE = [], [], []
    for test_size in [0.9, 0.7, 0.5, 0.3, 0.1]:
        R2.append([])
        RMSE.append([])
        MAE.append([])
        for num in tqdm(range(5)):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=2022 + num)
            # X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=2202 + num)
            model = RandomForestRegressor(oob_score=True, n_estimators=500)
            # X_train = np.concatenate([X_train, X_val], axis=0)
            # y_train = np.concatenate([y_train, y_val], axis=0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            R2[-1].append(r2_score(y_test, y_pred))
            RMSE[-1].append(mean_squared_error(y_test, y_pred) ** 0.5)
            MAE[-1].append(mean_absolute_error(y_test, y_pred))
            plt.scatter(y_test, y_pred, alpha=0.4, label='Predicted')
            plt.xlim(-5, 105)
            plt.ylim(-5, 105)
            plt.plot([-5, 105], [-5, 105])
            plt.xlabel('Observed Yield/%')
            plt.ylabel('Predicted Yield/%')
            plt.title(f'{method_name}_{test_size}\nR2: {round(R2[-1][-1], 4)}\nRMSE: {round(RMSE[-1][-1], 2)}\nMAE: {round(MAE[-1][-1], 2)}')
            plt.legend()
            plt.savefig(f'{method_name}_{test_size}.jpg')
            plt.clf()
    print(f'{method_name}' + '|  R2_score:', np.array(R2).mean(axis=0) , ' ± ' , np.array(R2).std(axis=0),
          ' RMSE:', np.array(RMSE).mean(axis=0) , ' ± ' , np.array(RMSE).std(axis=0),
          ' MAE:', np.array(MAE).mean(axis=0) , ' ± ' , np.array(MAE).std(axis=0))
    print(f'{method_name}' + '|  R2_score:', np.array(R2).mean(axis=1) , ' ± ' , np.array(R2).std(axis=1),
          ' RMSE:', np.array(RMSE).mean(axis=1) , ' ± ' , np.array(RMSE).std(axis=1),
          ' MAE:', np.array(MAE).mean(axis=1) , ' ± ' , np.array(MAE).std(axis=1)) 


training(Reaxtica_X, Y, 'Reaxtica')
training(drfp_X, Y, 'drfp')
training(rxnfp_X, Y, 'rxnfp')