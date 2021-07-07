from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import numpy as np

"""This is where the metrics are calculated
    """


def pr_auc(y, y_pred):
    """calculate the area under the precision recall curve"""
    p, r, _ = precision_recall_curve(y, y_pred)
    return auc(r, p)


def custom_metric(y, y_pred, fn_cost=0.72, fp_cost=0.28):
    """given labels and prediction, calculate the falase positive and false negative rate and adjust with the respective costs as weights"""
    fp = np.mean((y == 0) & (y_pred == 1))  # false positive rate
    fn = np.mean((y == 1) & (y_pred == 0))  # false negative rate
    score = -fp*fp_cost-fn*fn_cost
    return score
