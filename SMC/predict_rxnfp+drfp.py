import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from tqdm import trange


def predict_with_fps(fps_type, dataset):
    X = np.load(f'saved_descriptors/fps_{fps_type}_{dataset}.npy', allow_pickle=True)
    Y = pd.read_csv('datasets/USPTO_SMC.csv').rxn_yield.values if dataset == 'USPTO' else pd.read_csv('datasets/Reaxys_SMC.csv').rxn_yield.values
    R2, RMSE, MAE = [], [], []
    for num in trange(5):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2022 + num)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=2202 + num)
        X_train = np.concatenate([X_train, X_val], axis=0)
        y_train = np.concatenate([y_train, y_val], axis=0)
        model = XGBRegressor(n_estimators=999,
                learning_rate=0.01,
                max_depth=12,
                min_child_weight=6,
                colsample_bytree=0.6,
                subsample=0.8,
                random_state=42,)
        model.fit(
            X_train,
            y_train,
            verbose=False,)
        y_pred = model.predict(X_test)
        y_pred_val = model.predict(X_val)
        y_pred_train = model.predict(X_train)
        R2.append(r2_score(y_test, y_pred))
        RMSE.append(mean_squared_error(y_test, y_pred) ** 0.5)
        MAE.append(mean_absolute_error(y_test, y_pred))

    print('XGBoost' + '|  R2_score:', '%.4f' % np.array(R2).mean() + ' ± ' + '%.4f' % np.array(R2).std(),
          ' RMSE:', '%.3f' % np.array(RMSE).mean() + ' ± ' + '%.3f' % np.array(RMSE).std(),
          ' MAE:', '%.3f' % np.array(MAE).mean() + ' ± ' + '%.3f' % np.array(MAE).std())



predict_with_fps('drfp', 'Reaxys')
predict_with_fps('rxnfp', 'Reaxys')
predict_with_fps('drfp', 'USPTO')
predict_with_fps('rxnfp', 'USPTO')