import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

def find_optimal_cutoff(y, pred):
    """ Find the optimal cutoff for a binary classifier
    """
    fpr, tpr, threshold = roc_curve(y, pred)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])[0], auc(fpr, tpr)