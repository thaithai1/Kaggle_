import lightgbm
import numpy as np
import sys

def check_gpu_support():
    try:
        data = np.random.rand(50, 2)
        label = np.random.randint(2, size=50)
        train_data = lightgbm.Dataset(data, label=label)
        params = {'num_iterations': 1, 'device': 'gpu'}
        gbm = lightgbm.train(params, train_set=train_data)
        sys.stdout.write("True")
        return True
    except Exception as e:
        sys.stdout.write("False")
        return False
check_gpu_support()