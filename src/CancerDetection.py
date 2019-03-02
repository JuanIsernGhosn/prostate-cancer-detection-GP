import scipy.io
import numpy as np
from sklearn.model_selection import KFold
import random
import gpflow as gp

DATA_PATH = '../Datos.mat'

def __main__():
    data = read_data(DATA_PATH)
    gp_cross_val(data)


def gp_cross_val(data):

    negative_ex = data['Healthy_folds'].flatten()
    positive_ex = data['Malign_folds'].flatten()

    for i in range(0,5):
        train_index = range(0,5)
        del train_index[i]

        train_neg = np.vstack([negative_ex[i][0] for i in train_index])
        train_pos = np.vstack([positive_ex[i][0] for i in train_index])
        test_neg = negative_ex[i][0]
        test_pos = positive_ex[i][0]

        for part in partition(range(0,train_neg.shape[0]), 4):
            X = train_neg
            print("hola")
            #gp.models.SVGP(kern=gp.kernels.Linear)


def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def read_data(path):
    return scipy.io.loadmat(path)