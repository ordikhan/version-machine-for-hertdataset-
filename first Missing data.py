import pandas as pd
import numpy as np

df = pd.read_excel("prognosis.xlsx")
# print(df.head())

print(df.isnull().sum())