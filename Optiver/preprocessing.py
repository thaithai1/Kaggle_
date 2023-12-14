from data_manager import *

class Preprocessing_null:
    def run(self, X):
        return X.copy()

class Preprocessing_base(Preprocessing_null):
    def run(self, X):
        X_res = add_feat(X, prt=False)
        return X_res
    
class RestrictCols(Preprocessing_null):
    def __init__(self, cols = None) -> None:
        self.cols = cols

    def run(self, X):
        return X[self.cols].copy()
    
class ChainPrepro(Preprocessing_null):
    def __init__(self, prepro_iter) -> None:
        self.prepro_iter=prepro_iter
    
    def run(self, X):
        X_res = X.copy()
        for p in self.prepro_iter:
            X_res=p.run(X_res)
        return X_res
        