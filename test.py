
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
y_true = np.array([0.7, 0.8, 0.51, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
roc_auc_score(y_true, y_scores)
print(roc_auc_score(y_true, y_scores))