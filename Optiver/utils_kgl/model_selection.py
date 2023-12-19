import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import gc
import lightgbm
from lightgbm import Booster
if hasattr(Booster, '__deepcopy__'):
    del Booster.__deepcopy__
from copy import deepcopy
from utils_kgl.processing import *

def get_train_val_idx(num_folds, gap, date_ids):
    """date_ids has to start at 0 and be a continuous array"""
    max_date = len(np.unique(date_ids))-1
    split_idx = [i*max_date//num_folds for i in range(1,num_folds+1)]
    res = []
    for i in range(num_folds-1):
        start, end  = split_idx[i], split_idx[i+1]
        train_idx = date_ids < start
        test_idx = (date_ids >= start+gap) & (date_ids < end)
        res.append((train_idx,test_idx))
    return res

def verbose_score(scores, metrics = 'MAE'):
     for i in range(len(scores)):
          print(f"Fold {i+1} - {metrics}: {scores[i]}")

def score(model, df, target, metric = mean_absolute_error):
    predictions = model.predict(df)
    return metric(predictions, target)
          

def scoreCV(df, y, date_ids,model, prepro= Preprocessing_null(), num_folds = 5, gap = 5, verbose = False):
    models = []
    prepros = []
    scores = []
    cnt = 1
    for train_idx, test_idx in get_train_val_idx(num_folds, gap, date_ids):
        modelCV = deepcopy(model)
        preproCV = deepcopy(prepro)
        
        df_fold_train = df[train_idx]
        df_fold_train_target = y[train_idx]    
        df_fold_valid = df[test_idx]
        df_fold_valid_target = y[test_idx]

        # Features generation
        df_fold_train = preproCV.fit_transform(df_fold_train)
        df_fold_valid = preproCV.transform(df_fold_valid)

        gc.collect()

        if type(modelCV)==lgb.sklearn.LGBMRegressor:
            modelCV.fit(
                df_fold_train,
                df_fold_train_target,
                eval_set=[(df_fold_valid, df_fold_valid_target)],
                callbacks=[
                    lgb.callback.early_stopping(stopping_rounds=100),
                    lgb.callback.log_evaluation(period=100),
                ],
            )
        else : 
            modelCV.fit(
                df_fold_train,
                df_fold_train_target,
            )
        
        fold_score = score(modelCV, df_fold_valid, df_fold_valid_target, metric = mean_absolute_error)
        
        scores.append(fold_score)
        prepros.append(preproCV)
        models.append(modelCV)

        if verbose:
            print(f"Fold {cnt} - MAE: {fold_score}")
            cnt+=1

    return scores, models, prepros