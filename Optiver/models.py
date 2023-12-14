import numpy as np

class Model_base:
    def fit(self, X,y):
        pass

    def add_data(self):
        pass

    def predict(self,X):
        n = len(X)
        return np.zeros(n)
    

class Imbalanced(Model_base):
    def predict(self,X):
        n = len(X)
        d = {
            1:0.1,
            0:0,
            -1:-0.1
        }
        return X.imbalance_buy_sell_flag.map(d)

class Sklearn(Model_base):
    def __init__(self, skmodel):
        self.model = skmodel
    
    def fit(self, X, y):
        self.model.fit(X,y)
    
    def predict(self, X):
        return self.model.predict(X)
        

class Linear(Model_base):
    def __init__(self, cols) -> None:
        super().__init__()