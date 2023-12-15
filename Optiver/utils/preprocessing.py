from utils.data_manager import *

class Preprocessing_null:
    def fit_transform(self, X):
        return self.transform(X)

    def transform(self, X):
        return X.copy()

class Preprocessing_base(Preprocessing_null):
    def transform(self, X):
        X_res = add_feat(X, prt=False)
        return X_res
    
class RestrictCols(Preprocessing_null):
    def __init__(self, cols = None) -> None:
        self.cols = cols

    def transform(self, X):
        return X[self.cols].copy()
    
class ChainPrepro(Preprocessing_null):
    def __init__(self, prepro_iter) -> None:
        self.prepro_iter=prepro_iter
    
    def fit_transform(self, X):
        X_res = X.copy()
        for p in self.prepro_iter:
            print(p)
            X_res=p.fit_transform(X_res)
        return X_res

    def transform(self, X):
        X_res = X.copy()
        for p in self.prepro_iter:
            X_res=p.transform(X_res)
        return X_res
    
class PreproSk(Preprocessing_null):
    def __init__(self, sklearn):
        self.sklearn= sklearn
    
    def fit_transform(self,X):
        return self.sklearn.fit_transform(X)
    
    def transform(self,X):
        return self.sklearn.transform(X)
        