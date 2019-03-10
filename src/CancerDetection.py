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
    # Load data
    neg_fold, pos_fold = read_mat(DATA_PATH)
    # Normalize data (Z-Score)
    neg_fold_norm, pos_fold_norm = norm_data(neg_fold, pos_fold)

    # Compute GP with Gaussian kernel (RBF)
    roc_gaussian = gp_cross_val(neg_fold_norm, pos_fold_norm, krnl="Gaussian")
    write_data(roc_gaussian, "auc_gaussian.pkl")

    # Compute GP with Linear kernel (RBF)
    roc_linear = gp_cross_val(neg_fold_norm, pos_fold_norm, krnl="Linear")
    write_data(roc_linear, "auc_linear.pkl")

    # Read data (If is already generated)
    roc_gaussian = read_data("auc_gaussian.pkl")
    roc_linear = read_data("auc_linear.pkl")

    # ---- Gaussian Kernel ----
    # Plot confusion matrix
    plot_cm_grid(roc_gaussian.get("cm"), title='Núcleo Gaussiano')
    # Plot ROC curves
    plot_roc(roc_gaussian.get('one_minus_specificity'), roc_gaussian.get('recall_1'),
             roc_gaussian.get('one_m_spe_recall_auc'), title='ROC Núcleo Gaussiano',
             x_label='1 - Especificidad', y_label="Sensibilidad")
    # Plot precision/recall curves
    plot_roc(roc_gaussian.get('recall_2'), roc_gaussian.get('precision'),
             roc_gaussian.get('prec_rec_auc'), title='Precisión-Sensibilidad Núcleo Gaussiano',
             x_label="Sensibilidad", y_label="Precisión")
    # Plot metrics table
    print(plot_metrics_table(cms=roc_gaussian.get("cm")))

    # ---- Linear Kernel ----
    # Plot confusion matrix
    plot_cm_grid(roc_linear.get("cm"), title='Núcleo Lineal')
    # Plot ROC curves
    plot_roc(roc_linear.get('one_minus_specificity'), roc_linear.get('recall_1'),
             roc_linear.get('one_m_spe_recall_auc'), title='ROC Núcleo Lineal',
             x_label='1 - Especificidad', y_label="Sensibilidad")
    # Plot precision/recall curves
    plot_roc(roc_linear.get('recall_2'), roc_linear.get('precision'),
             roc_linear.get('prec_rec_auc'), title='Precisión-Sensibilidad Núcleo Lineal',
             x_label="Sensibilidad", y_label="Precisión")
    # Plot metrics table
    print(plot_metrics_table(cms=roc_linear.get("cm")))


def plot_metrics_table(cms):
    """Plot table with evaluation metrics for given confusion matrix list.   
    Compute Accuracy, precision, recall, specificity, F1 and save into a dataframe   
    Args:
        (cm[]) cms: List of confusion matrix
    Returns:
        (DataFrame): Scores for each CV split.
    """
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
    return (pd.DataFrame(data=d))


def plot_cm_grid(cms, title):
    """Plot a grid of confusion matrix.  
    Args:
        (cm[]) cms: List of confusion matrix to plot
        (String) title: Main title of the grid
    """
    size = 321
    plt.figure(figsize=(8, 8))
    plt.suptitle('Matrices de confusión: ' + title)
    for i, cm in enumerate(cms):
        plt.subplot(size + i)
        # Plot each confusion matrix
        plot_confusion_matrix(cm, target_names=['Sana', 'Cáncer'], title='Fold ' + str(i + 1))
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()


def plot_confusion_matrix(cm, target_names=['0', '1'], title='Confusion matrix', normalize=False):
    """Plot confusion matrix with better aesthetic.
    Args:
        (int[][]) cm: Confusion matrix to plot.
        (String[]) target_names: Name of the matrix labels
        (String) title: Title to print above the matrix.
        (bool) normalize: Set if is matrix normalization required.
    """
    accuracy = np.trace(cm) / float(np.sum(cm))

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
    """Plot ROC / specificity-recall curve.
    Args:
        (float[][]) x: x coord points for each CV split.
        (float[][]) y: y coord points for each CV split.
        (float[]) roc_auc: Area under curve made with x and y points
        (String) title: Title to print above the curve.
        (String) x_label: X axis label.
        (String) y_label: Y axis label.
    """
    plt.figure()
    for i in list(range(len(x))):
        plt.plot(x[i], y[i], label='Fold {0} (área = {1:0.2f})'.format(i + 1, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="lower right")
    plt.title('Curvas ' + title)
    plt.show()


def norm_data(neg_fold, pos_fold):
    """Normalize positive and negative folds data
    Args:
        (float[][][]) neg_fold: negative folds data
        (float[][][]) pos_fold: positive folds data
    Returns:
        (float[][][]): negative normalized folds data
        (float[][][]): positive normalized folds data
    """
    orig_index_n = get_fold_index(neg_fold)
    join_neg = np.vstack([fold for fold in neg_fold])
    orig_index_p = get_fold_index(pos_fold, ini=(join_neg.shape[0]))
    join_pos = np.vstack([fold for fold in pos_fold])

    join = np.concatenate((join_neg, join_pos))
    join_n = preprocessing.scale(join, axis=0)

    return [join_n[fold_idx] for fold_idx in orig_index_n], [join_n[fold_idx] for fold_idx in orig_index_p]


def get_fold_index(folds, ini=0):
    """Get folds data indexes
    Args:
        (float[][][]) folds: CV folds data
    Returns:
        (int[][]): indexes for each fold
    """
    orig_index = []
    for n, fold in enumerate(folds):
        ini = ini + folds[n - 1].shape[0] if n > 0 else ini
        orig_index.append(list(range(ini, ini + fold.shape[0])))
    return orig_index


def gp_cross_val(neg_fold, pos_fold, krnl="Linear", threshold=0.5):
    """Make CV of Gaussian Process with given kernel and compute ROC
    and specificity-recall curve for each split.
    Args:
        (float[][][]) neg_fold: negative folds data
        (float[][][]) pos_fold: positive folds data
        (String) krnl: Kernel of the Gaussian process
        (float) threshold: threshold for translating predicted probabilities into classes   
    Returns:
        (dictionary): Curve points and AUC information.
    """

    recall_1, recall_2, one_m_spec, spec_rec_auc, prec, prec_rec_auc, cms  = ([] for _ in range(7))

    # 5-fold Cross Validation
    for i in range(0, 5):
        # Select folds for current data split
        train_index = list(range(0, 5))
        train_index.remove(i)

        train_neg = np.vstack([neg_fold[j] for j in train_index])
        train_pos = np.vstack([pos_fold[j] for j in train_index])

        test_neg = neg_fold[i]
        test_pos = pos_fold[i]

        x_test = np.concatenate((test_neg, test_pos))
        y_test = np.append([-1] * test_neg.shape[0], np.ones(test_pos.shape[0]))

        probs_fold = []

        # Models for balanced classification into fold (All positive examples vs 1/4 negative fold data)
        for part in partition(list(range(0, train_neg.shape[0])), 4):
            x = np.concatenate((train_neg[part], train_pos))
            y = np.append([-1] * len(part), np.ones(len(train_pos)))
            y = y.reshape(-1, 1)

            idx = np.random.permutation(len(x))
            x, y = x[idx], y[idx]

            # Kernel selection
            if krnl == "Linear":
                m = gp.models.VGP(X=x, Y=y, kern=gp.kernels.Linear(x.shape[1]), likelihood=gp.likelihoods.Bernoulli())
            else:
                m = gp.models.VGP(X=x, Y=y, kern=gp.kernels.RBF(x.shape[1]), likelihood=gp.likelihoods.Bernoulli())

            # Model optimization
            gp.train.ScipyOptimizer().minimize(m, maxiter=300)

            # Prediction with test data
            p = m.predict_y(x_test)[0]

            probs_fold.append(p)

        # Mean of the probability of the 4 models prediction for current CV split
        mean_probs = np.mean(probs_fold, axis=0).flatten()

        # ROC curve and AUC calculation
        one_m_spec_v, recall_v, thresholds = metrics.roc_curve(y_test, mean_probs)
        one_m_spec.append(one_m_spec_v)
        recall_1.append(recall_v)
        spec_rec_auc.append(metrics.auc(one_m_spec_v, recall_v))

        # Recall-specificity curve and AUC calculation
        prec_v, recall_v, thresholds = metrics.precision_recall_curve(y_test, mean_probs)
        prec.append(prec_v)
        recall_2.append(recall_v)
        prec_rec_auc.append(metrics.auc(recall_v, prec_v))

        # Confusion matrix calculation
        y_pred = [1 if prob > threshold else -1 for prob in mean_probs]
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
        cPickle.dump(data, fid, protocol=2)


def partition(list_in, n):
    """Make n-random partitions from list
    Args:
        (list) list_in: List to be partitioned.
        (int) n: Number of partitions.
    Returns:
        (type[][]): Random partitions
    """
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def read_mat(path):
    """Read .mat file.
    Args:
        (String) path: Path to the .mat file to read
    Returns:
        (float[][][]): negative folds data
        (float[][][]): positive folds data
    """
    mat = io.loadmat(path)
    negative_ex = mat['Healthy_folds'].flatten()
    positive_ex = mat['Malign_folds'].flatten()
    negative_ex = [negative_ex[i][0] for i in list(range(0, 5))]
    positive_ex = [positive_ex[i][0] for i in list(range(0, 5))]
    return negative_ex, positive_ex