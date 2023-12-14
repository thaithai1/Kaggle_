from data_manager import *

class Preprocessing_base:
    def preprocess(self, df):
        df_res = add_feat(df, prt=False)
        return df_res