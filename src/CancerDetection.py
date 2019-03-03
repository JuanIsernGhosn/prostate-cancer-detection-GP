import scipy.io
import numpy as np
from sklearn import metrics
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
        train_index = list(range(0,5))
        train_index.remove(i)

        train_neg = np.vstack([negative_ex[i][0] for i in train_index])
        train_pos = np.vstack([positive_ex[i][0] for i in train_index])

        test_neg = negative_ex[i][0]
        test_pos = positive_ex[i][0]
        X_test = np.concatenate((test_neg, test_pos))
        Y_test = np.append([-1] * test_neg.shape[0], np.ones(test_pos.shape[0]))


        for part in partition(list(range(0,train_neg.shape[0])), 4):
            X = np.concatenate((train_neg[part],train_pos))
            Y = np.append([-1] * len(part),np.ones(len(train_pos)))
            Y.reshape(-1, 1)

            idx = np.random.permutation(len(X))
            X, Y = X[idx], Y[idx]

            m = gp.models.SVGP(X=X, Y=Y, kern=gp.kernels.Linear, likelihood=gp.likelihoods.Gaussian)
            m.feature.set_trainable(False)
            gp.train.ScipyOptimizer().minimize(m, maxiter=20)
            m.feature.set_trainable(True)
            gp.train.ScipyOptimizer(options=dict(maxiter=200)).minimize(m)


            p = m.predict_y(X_test)

            print("hola")




def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def read_data(path):
    return scipy.io.loadmat(path)