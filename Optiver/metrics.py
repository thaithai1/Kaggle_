from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
from tqdm import tqdm

def score(df_train_raw, df_test_raw, model, preprocess):
    data_res = []
    
    #Model setup
    df_train = preprocess.preprocess(df_train_raw)
    model.fit(df_train)

    df_test_raw.sort_values(by=['date_id', 'stock_id', 'seconds_in_bucket'], inplace=True)
    for date in tqdm(np.sort(df_test_raw.date_id.unique())):
        df_date = df_test_raw[df_test_raw.date_id==date]
        for sec in range(0, 541, 10):

            #Data setup
            test = df_date[df_date['seconds_in_bucket']==sec].copy()
            y = test['target']
            test.drop('target', axis=1, inplace=True)
            sample_prediction = pd.DataFrame({'row_id':test.row_id, 'target':np.nan})

            #Submission model
            X_test = preprocess.preprocess(test)
            model.add_data()
            sample_prediction['target'] = model.predict(X_test)


            #Performance assessment
            df_res_temp = sample_prediction.copy()
            df_res_temp.rename(columns={'target':'pred'}, inplace=True)
            df_res_temp['target'] = y
            data_res.append(df_res_temp)

    df_res = pd.concat(data_res)
    mae = mean_absolute_error(df_res.target,df_res.pred)
    return df_res, mae


