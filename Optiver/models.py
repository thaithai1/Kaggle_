import numpy as np

class Model_base:
    def fit(self, X_train):
        pass
    def add_data(self):
        pass
    def predict(self,test):
        n = len(test)
        return np.zeros(n)