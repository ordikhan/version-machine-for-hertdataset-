import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random
import pickle


import numpy as np
from sklearn.model_selection import KFold

dataframe = pd.read_excel('edit.xlsx',
                                   names=["IHHPCode", "sex", "whr", "smoker", "family_history", "diabetes", "age",
                                          "cholesterol", "blood_pressure", "dbp", "hdl", "ldl", "tg", "htn",
                                          "sbp_cat", "tchcat", "ch1", "ch2", "ch3", "ch4", "ch5", "bp1",
                                          "bp2", "bp3", "bp4", "FollowDu5th", "label"], header=0)
X = df.iloc[:, 0:16]
Y = df.iloc[:, 17]

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)
kf.get_n_splits(X)

print(kf)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]





random.seed(2002)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3)



dataframe = pd.read_excel('edit.xlsx',
                                   names=["IHHPCode", "sex", "whr", "smoker", "family_history", "diabetes", "age",
                                          "cholesterol", "blood_pressure", "dbp", "hdl", "ldl", "tg", "htn",
                                          "sbp_cat", "tchcat", "ch1", "ch2", "ch3", "ch4", "ch5", "bp1",
                                          "bp2", "bp3", "bp4", "FollowDu5th", "label"], header=0)
X = dataframe.loc[:, {"sex", "whr", "smoker", "family_history", "diabetes", "age", "cholesterol", "blood_pressure","label"}]
SVM = SVC(kernel='rbf', gamma=0.2, C=1.8)  # sigmoid rbf poly linear
SVM = SVM.fit(Xtrain, Ytrain)

pred = SVM.predict(Xtest)
acc = accuracy_score(Ytest, pred)
print(acc)

print(SVM.support_vectors_)
print(SVM.support_)
print(SVM.n_support_)

save_classifier = open('C:/Users/e.almaee/Desktop/Dataset/svm.pickle', 'wb')
pickle.dump(SVM, save_classifier)
save_classifier.close()


