import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import random
from sklearn.ensemble import VotingClassifier


random.seed(2002)

import pandas as pd

# define dataset
df = pd.read_excel("heart.xlsx")
print(df.head())
data = df.to_numpy()
X = data[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
y = data[:, -1]

tree = DecisionTreeClassifier()
GNB = GaussianNB()
BNB = BernoulliNB()

vote = VotingClassifier(estimators=[('tree',tree),('Gnb', GNB),('Bnb', BNB)], weights=[2,1,1])
vote.fit(X,y)
pred = vote.predict(X)

print(accuracy_score(y, pred))