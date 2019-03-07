import scipy.io
import numpy as np
from sklearn import preprocessing, metrics
import random
import gpflow as gp
import _pickle as cPickle
from itertools import cycle
import matplotlib.pyplot as plt

DATA_PATH = '../Datos.mat'

def __main__():
    '''
    neg_fold, pos_fold = read_mat(DATA_PATH)
    neg_fold_norm = norm_data(neg_fold)
    pos_fold_norm = norm_data(pos_fold)

    auc = gp_cross_val(neg_fold_norm, pos_fold_norm)

    write_data(auc, "auc_linear.pkl")
    '''



    fpr, tpr, tresholds = read_data("auc_linear.pkl")

    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    #plot_roc(fpr, tpr)


def plot_roc(fpr_folds, tpr_folds):

    plt.figure()

    for fpr, tpr in zip(fpr_folds, tpr_folds ):
        plt.plot(fpr, tpr)

    plt.show()


def norm_data(folds):
    return [preprocessing.normalize(fold, axis=0) for fold in folds]

def gp_cross_val(neg_fold, pos_fold):

    tpr = []
    fpr = []
    roc_auc = []

    for i in range(0,5):
        train_index = list(range(0,5))
        train_index.remove(i)

        train_neg = np.vstack([neg_fold[j] for j in train_index])
        train_pos = np.vstack([pos_fold[j] for j in train_index])

        test_neg = neg_fold[i]
        test_pos = pos_fold[i]

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

        fpr_v, tpr_v, thresholds = metrics.roc_curve(y_test, mean_probs)
        tpr.append(tpr_v)
        fpr.append(fpr_v)
        roc_auc.append(metrics.auc(fpr_v, tpr_v))

    return (fpr, tpr, roc_auc)


def read_data(path):
    """Read .pkl file saved locally.

    Args:
        (String) path: Path of the file to load.
    Returns:
        (Object): Read file object.

    """
    with open(path, 'rb') as fid:
        return cPickle.load(fid)


def write_data(data, path):
    """Save .pkl file locally.

    Args:
        (Object): Object to be saved.
    """
    with open(path, 'wb') as fid:
        cPickle.dump(data, fid)

def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def read_mat(path):
    mat = scipy.io.loadmat(path)
    negative_ex = mat['Healthy_folds'].flatten()
    positive_ex = mat['Malign_folds'].flatten()
    negative_ex = [negative_ex[i][0] for i in list(range(0,5))]
    positive_ex = [positive_ex[i][0] for i in list(range(0,5))]
    return negative_ex, positive_ex