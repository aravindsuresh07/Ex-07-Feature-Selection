
Developed By: Aravind S
Reg.No: 212220220004

# PROGRAM FOR TITANIC


import pandas as pd
import numpy as np
df = pd.read_csv("titanic_dataset.csv")
df
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df[['Age']] = imputer.fit_transform(df[['Age']])
print("Feature selection")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
selector = SelectKBest(chi2, k=3)
X_new = selector.fit_transform(X, y)
print(X_new)
df_new = pd.DataFrame(X_new, columns=['Pclass', 'Age', 'Fare'])
df_new['Survived'] = y.values
df_new.to_csv('titanic_transformed.csv', index=False)
print(df_new)
