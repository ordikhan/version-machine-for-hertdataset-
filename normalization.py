from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale

import pandas as pd

# define dataset
df = pd.read_excel("heart.xlsx")
print(df.head())
data = df.to_numpy()
X = data[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
y = data[:, -1]

Xnorm = MinMaxScaler().fit_transform(X)

print(X)
print(Xnorm)

Xscale = scale(X)
print(Xscale)

