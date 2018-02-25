# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 21:16:29 2018

@author: HatemZam
"""

import pandas as pd
import numpy as np

dataFrame = pd.read_csv('D:/3rd Year Comp. Eng/2nd Term/AI/sections/Abalone/abalone.csv')
#columnsNames = pd.read_csv('D:/3rd Year Comp. Eng/2nd Term/AI/sections/Abalone/abalone (2).csv')
df = pd.DataFrame()
#df = pd.DataFrame(columns= ['sex','length','diameter','height','whole weight','shucked weight','viscera weight','shell weight','rings'])

#s = pd.Series(['M', 0.455, 0.365, 0.095, 0.514, 0.2245, 0.101, 0.15, 15])

df2 = df.append(dataFrame, ignore_index=True)
df2.columns = ['sex','length','diameter','height','whole weight','shucked weight','viscera weight','shell weight','rings']

#df3 = df.append(dataFrame)

#----------------------------------Convert the Categorical feature string to values

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df2.sex)

df2.sex = le.transform(df2.sex)

#----------------------------------Split the features from the Dependent Var.
X = df2.iloc[:, :8].values
y = df2.iloc[:, 8].values
#----------------------------------Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
#----------------------------------Splitting the data into Train Set and Test Set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#----------------------------------FiTTing The Random Forest Regression

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 5000, random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# --------------------------------- Calculate RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
sqrt(mean_squared_error(y_test, y_pred))
#Or
sqrt(np.mean((y_test-y_pred)**2))











