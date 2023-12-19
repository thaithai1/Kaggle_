import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

path = 'data/optiver-trading-at-the-close/'

def get_train(local_path=None, dropNull=True):
    if local_path is None:
        p = path+'train.csv'
    else:
        p = local_path+'train.csv'
    print(f"{p} - loaded")
    df = pd.read_csv(p, index_col='row_id')
    if dropNull:
        df=df[~df.target.isnull()]
        df=df[~df.wap.isna()] # when missing, the whole date_id / stock_id missing
    return df

def get_test():
    p =path+'example_test_files/test.csv'
    print(f"{p} - loaded")
    return pd.read_csv(p, index_col='row_id')

def add_feat(df, prt=True):
    df_res = df.copy()
    feat = ['imbalance_size_rel','wap60', 'wap_ret', 'imb/volume']
    if prt:
        print(f"{', '.join(feat)} - added")
    df_res['imbalance_size_rel'] = df_res['imbalance_size'] * df_res['imbalance_buy_sell_flag']
    df_res.sort_values(by=['stock_id', 'date_id', 'seconds_in_bucket'], inplace=True)
    df_res['wap60'] = df_res.groupby(['stock_id', 'date_id'])['wap'].shift(-6)
    df_res['wap_ret'] = (df_res['wap60'] / df_res['wap']-1)*1e4
    df_res['imb/volume'] = df_res['imbalance_size_rel']/(df_res['ask_size']+df_res['bid_size'])
    return df_res

def get_df_id_date(df, id, date_id): 
    return df[(df["stock_id"]==id) & (df["date_id"]==date_id)]

def plot_id_date(df, id, date_id):

    df = get_df_id_date(df, id, date_id)

    fig, ax = plt.subplots(4,1, figsize=(14, 8))

    #Price
    sns.lineplot(data=df, x="seconds_in_bucket", y="wap", label="WAP", ax=ax[0])
    sns.lineplot(data=df, x="seconds_in_bucket", y="wap60", label="WAP60", ax=ax[0], linestyle='--', color = 'darkblue', linewidth=1, alpha = 0.5)
    ax[0].fill_between(df["seconds_in_bucket"], df["bid_price"], df["ask_price"], alpha=0.2, label="Bid-Ask Spread")
    sns.lineplot(data=df, x="seconds_in_bucket", y="reference_price", label ="reference", ax=ax[0])
    sns.lineplot(data=df, x="seconds_in_bucket", y="near_price", label = "near", ax=ax[0])
    sns.lineplot(data=df, x="seconds_in_bucket", y="far_price", label = "far", ax=ax[0])
    ax[0].axvline(x=300, linestyle=':', linewidth = 1)

    # Auction volume
    sns.lineplot(data=df, x="seconds_in_bucket", y="matched_size", label ="matched_size", ax = ax[1])
    sns.lineplot(data=df, x="seconds_in_bucket", y="imbalance_size_rel", label ="relative imbalance", ax = ax[1])
    ax[1].axvline(x=300, linestyle=':', linewidth = 1)
    ax[1].axhline(y=0, linestyle='--', linewidth = 1)
    ax[1].legend()

    # Order book pressure
    sns.lineplot(data=df, x="seconds_in_bucket", y="ask_size", label ="ask size", ax = ax[2])
    ax[2].plot(df.seconds_in_bucket,-df.bid_size, linewidth = 1, label = "bide")
    ax[2].axvline(x=300, linestyle=':', linewidth = 1)
    ax[2].axhline(y=0, linestyle='--', linewidth = 1)

    # Target
    sns.lineplot(data=df, x="seconds_in_bucket", y="target", label ="target", ax = ax[3])
    sns.lineplot(data=df, x="seconds_in_bucket", y="wap_ret", label ="wap_ret", ax = ax[3])
    ax[3].axvline(x=300, linestyle=':', linewidth = 1)

def split_by_date(df, split_ratio):
    split_ratio = 0.2
    n = len(df.date_id.unique())
    idx_split = math.floor(n*(1-split_ratio))
    df_train = df[df['date_id']<=idx_split].copy()
    df_test = df[df['date_id']>idx_split].copy()
    return df_train, df_test
