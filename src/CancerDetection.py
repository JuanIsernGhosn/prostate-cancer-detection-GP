import scipy.io
import numpy as np
from sklearn.metrics import roc_curve, auc
import random
import gpflow as gp

DATA_PATH = '../Datos.mat'

def __main__():
    data = read_data(DATA_PATH)
    gp_cross_val(data)


def gp_cross_val(data):

    negative_ex = data['Healthy_folds'].flatten()
    positive_ex = data['Malign_folds'].flatten()

    tpr = []
    fpr = []
    roc_auc = []

    for f, i in enumerate(range(0,5)):
        train_index = list(range(0,5))
        train_index.remove(i)

        train_neg = np.vstack([negative_ex[i][0] for i in train_index])
        train_pos = np.vstack([positive_ex[i][0] for i in train_index])

        test_neg = negative_ex[i][0]
        test_pos = positive_ex[i][0]
        x_test = np.concatenate((test_neg, test_pos))
        y_test = np.append([-1] * test_neg.shape[0], np.ones(test_pos.shape[0]))

        probs_fold = []

        for part in partition(list(range(0,train_neg.shape[0])), 4):
            x = np.concatenate((train_neg[part],train_pos))
            y = np.append([-1] * len(part),np.ones(len(train_pos)))
            y = y.reshape(-1, 1)

            idx = np.random.permutation(len(x))
            x, y = x[idx], y[idx]

            m = gp.models.VGP(X=x, Y=y, kern=gp.kernels.RBF(x.shape[1]), likelihood=gp.likelihoods.Bernoulli())

            gp.train.ScipyOptimizer().minimize(m, maxiter=300)

            p = m.predict_y(x_test)[0]

            probs_fold.append(p)

        mean_probs = np.mean(probs_fold, axis=0).flatten()

        tpr_v, fpr_v = roc_curve(y_test, mean_probs, 1)
        tpr.append(tpr_v)
        fpr.append(fpr_v)
        roc_auc.append(auc(fpr_v, tpr_v))

        print(mean_probs)




def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def read_data(path):
    return scipy.io.loadmat(path)