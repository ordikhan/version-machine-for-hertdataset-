import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

y_pred=[1,2,2,3]

y_test=[0,0,1,1]
fpr, tpr, _ = roc_curve(y_test, y_pred)
auc_score = auc(fpr, tpr)
print(auc_score)