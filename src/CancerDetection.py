from scipy import io, stats
import numpy as np
from sklearn import metrics, preprocessing
import random
import gpflow as gp
import _pickle as cPickle
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd

DATA_PATH = '../Datos.mat'

def __main__():
    '''
    # Load data
    neg_fold, pos_fold = read_mat(DATA_PATH)
    # Normalize data (Z-Score)
    neg_fold_norm, pos_fold_norm = norm_data(neg_fold, pos_fold)
    
    
    # Compute GP with Gaussian kernel (RBF)
    roc_gaussian = gp_cross_val(neg_fold_norm, pos_fold_norm, krnl="Gaussian")
    #write_data(roc_gaussian, "auc_gaussian.pkl")
    
    
    # Compute GP with Linear kernel (RBF)
    roc_linear = gp_cross_val(neg_fold_norm, pos_fold_norm, krnl="Linear")
    #write_data(roc_linear, "auc_linear.pkl")
    '''

    roc_gaussian = read_data("auc_gaussian.pkl")
    roc_linear = read_data("auc_linear.pkl")

    # Plot ROC curves
    plot_roc(roc_gaussian.get('one_minus_specificity'), roc_gaussian.get('recall_1'),
             roc_gaussian.get('one_m_spe_recall_auc'), title = 'ROC Núcleo Gaussiano',
             x_label = '1 - Especificidad', y_label = "Sensibilidad")
    # Plot precision - recall curves
    plot_roc(roc_gaussian.get('recall_2'), roc_gaussian.get('precision'),
             roc_gaussian.get('prec_rec_auc'), title='ROC Núcleo Gaussiano',
             x_label="Sensibilidad", y_label="Precisión")


    #plot_cm_grid(roc_gaussian.get("cm"), title='Núcleo Gaussiano')
    #plot_metrics_table(cms = auc_linear.get("cm"), title='Núcleo Gaussiano')

def plot_metrics_table(cms, title):
    # Define metrics dictionary
    d = {'Accuracy': [], 'Specificity': [], 'Recall': [], 'Precision': [], 'F_Score': []}
    # Iterate over confusion matrix set
    for cm in cms:
        # Get confusion matrix elements and compute metrics
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        d.get('Accuracy').append((tp + tn) / (tp + fn + fp + tn))
        d.get('Specificity').append(tn / (tn + fp))
        d.get('Recall').append(recall)
        d.get('Precision').append(precision)
        d.get('F_Score').append((2 * precision * recall) / (precision + recall))
    return(pd.DataFrame(data=d))

def plot_cm_grid(cms, title):
    size = 321
    plt.figure(figsize=(10, 10))
    plt.suptitle('Matrices de confusión: '+title)
    for i, cm in enumerate(cms):
        plt.subplot(size + i)
        plot_confusion_matrix(cm, target_names=['Sana', 'Cáncer'], title='Fold '+str(i+1))
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

def plot_confusion_matrix(cm, target_names=['0', '1'], title='Confusion matrix',normalize=False):
    """Plot confusion matrix with better aesthetic.
    Args:
        (int[][]) cm: Confusion matrix to plot.
        (String[]) target_names: Name of the matrix labels
        (String) title: Title to print above the matrix.
        (bool) normalize: Set if is matrix normalization required.
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Etiqueta real')
    plt.xlabel('Etiqueta predicha')

def plot_roc(x, y, roc_auc, title, x_label, y_label):
    plt.figure()
    for i in list(range(len(x))):
        plt.plot(x[i], y[i], label='Fold {0} (área = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="lower right")
    plt.title('Curva '+title)
    plt.show()


def norm_data(folds_neg, folds_pos):
    orig_index_n = get_fold_index(folds_neg)
    join_neg = np.vstack([fold for fold in folds_neg])
    orig_index_p = get_fold_index(folds_pos, ini=(join_neg.shape[0]))
    join_pos = np.vstack([fold for fold in folds_pos])

    join = np.concatenate((join_neg, join_pos))
    join_n = preprocessing.scale(join, axis=0)

    return [join_n[fold_idx] for fold_idx in orig_index_n], [join_n[fold_idx] for fold_idx in orig_index_p]


def get_fold_index(folds, ini=0):
    orig_index = []
    for n, fold in enumerate(folds):
        ini = ini + folds[n - 1].shape[0] if n > 0 else ini
        orig_index.append(list(range(ini, ini + fold.shape[0])))
    return orig_index


def gp_cross_val(neg_fold, pos_fold, krnl = "Linear", treshold = 0.5):
    recall_1 = []
    recall_2 = []
    one_m_spec = []
    spec_rec_auc = []
    prec = []
    prec_rec_auc = []
    cms = []

    for i in range(0, 5):
        train_index = list(range(0, 5))
        train_index.remove(i)

        train_neg = np.vstack([neg_fold[j] for j in train_index])
        train_pos = np.vstack([pos_fold[j] for j in train_index])

        test_neg = neg_fold[i]
        test_pos = pos_fold[i]

        x_test = np.concatenate((test_neg, test_pos))
        y_test = np.append([-1]*test_neg.shape[0], np.ones(test_pos.shape[0]))

        probs_fold = []

        for part in partition(list(range(0, train_neg.shape[0])), 4):
            x = np.concatenate((train_neg[part], train_pos))
            y = np.append([-1]*len(part), np.ones(len(train_pos)))
            y = y.reshape(-1, 1)

            idx = np.random.permutation(len(x))
            x, y = x[idx], y[idx]

            if krnl == "Linear":
                m = gp.models.VGP(X=x, Y=y, kern=gp.kernels.Linear(x.shape[1]), likelihood=gp.likelihoods.Bernoulli())
            else:
                m = gp.models.VGP(X=x, Y=y, kern=gp.kernels.RBF(x.shape[1]), likelihood=gp.likelihoods.Bernoulli())

            gp.train.ScipyOptimizer().minimize(m, maxiter=300)

            p = m.predict_y(x_test)[0]

            probs_fold.append(p)

        mean_probs = np.mean(probs_fold, axis=0).flatten()

        one_m_spec_v, recall_v, thresholds = metrics.roc_curve(y_test, mean_probs)
        one_m_spec.append(one_m_spec_v)
        recall_1.append(recall_v)
        spec_rec_auc.append(metrics.auc(one_m_spec_v, recall_v))

        prec_v, recall_v, thresholds = metrics.precision_recall_curve(y_test, mean_probs)
        prec.append(prec_v)
        recall_2.append(recall_v)
        prec_rec_auc.append(metrics.auc(recall_v, prec_v))

        y_pred = [1 if prob > treshold else -1 for prob in mean_probs]

        cms.append(metrics.confusion_matrix(y_test, y_pred))

    return {"one_minus_specificity": one_m_spec, "recall_1": recall_1, "one_m_spe_recall_auc": spec_rec_auc,
            "precision": prec, "recall_2": recall_2, "prec_rec_auc": prec_rec_auc, "cm": cms}


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


def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def read_mat(path):
    mat = io.loadmat(path)
    negative_ex = mat['Healthy_folds'].flatten()
    positive_ex = mat['Malign_folds'].flatten()
    negative_ex = [negative_ex[i][0] for i in list(range(0, 5))]
    positive_ex = [positive_ex[i][0] for i in list(range(0, 5))]
    return negative_ex, positive_ex
