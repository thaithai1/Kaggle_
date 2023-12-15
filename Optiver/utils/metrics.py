from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.preprocessing import *

def score(df_res, metric_func=mean_absolute_error):
    return metric_func(df_res.target,df_res.pred)

def train_test_score(df_train_raw, df_test_raw, model, preprocess=Preprocessing_null()):
    data_res = []
    
    # Model setup
    X_train_raw, y_train = df_train_raw.drop('target', axis=1), df_train_raw['target']
    X_train = preprocess.fit_transform(X_train_raw)
    model.fit(X_train, y_train)

    # Train score
    df_res_train = pd.DataFrame({'target':y_train}, index = X_train.index)
    df_res_train['pred'] = model.predict(X_train)
    mae_train = score(df_res_train)

    # Test score
    df_test_raw.sort_values(by=['date_id', 'stock_id', 'seconds_in_bucket'], inplace=True)
    for date in tqdm(np.sort(df_test_raw.date_id.unique())):
        df_date = df_test_raw[df_test_raw.date_id==date]
        for sec in range(0, 541, 10):

            # Data setup
            test = df_date[df_date['seconds_in_bucket']==sec].copy()
            y_test = test['target']
            test.drop('target', axis=1, inplace=True)
            sample_prediction = pd.DataFrame({'target':np.nan}, index = test.index)

            # Submission model
            X_test = preprocess.transform(test)
            model.add_data() #TO MODIFY =======
            sample_prediction['target'] = model.predict(X_test)


            # Performance assessment
            df_res_temp = sample_prediction.copy()
            df_res_temp.rename(columns={'target':'pred'}, inplace=True)
            df_res_temp['target'] = y_test
            data_res.append(df_res_temp)

    df_res_test = pd.concat(data_res)
    mae_test = score(df_res_test)
    return df_res_train, df_res_test, mae_train, mae_test


def train_test_score_fast(df_train_raw, df_test_raw, model, preprocess=Preprocessing_null()):
    data_res = []
    
    # Model setup
    X_train_raw, y_train = df_train_raw.drop('target', axis=1), df_train_raw['target']
    X_test_raw, y_test = df_test_raw.drop('target', axis=1), df_test_raw['target']
    X_train = preprocess.fit_transform(X_train_raw)
    X_test = preprocess.transform(X_test_raw)
    model.fit(X_train, y_train)

    # Train score
    df_res_train = pd.DataFrame({'target':y_train}, index = X_train.index)
    df_res_train['pred'] = model.predict(X_train)
    mae_train = score(df_res_train)
    # Test score
    df_res_test = pd.DataFrame({'target':y_test}, index = X_test.index)
    df_res_test['pred'] = model.predict(X_test)
    mae_test = score(df_res_test)

    return df_res_train, df_res_test, mae_train, mae_test

def score_CV():
    pass

