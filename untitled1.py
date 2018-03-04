# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:55:38 2018

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

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
le.fit(df2.sex)
df2.sex = le.transform(df2.sex)

onehotencoder = OneHotEncoder(categorical_features = [0])
df2 = onehotencoder.fit_transform(df2).toarray()

#----------------------------------Split the features from the Dependent Var.
X = df2[:, :10]
y = df2[:, 10]

#----------------------------------Splitting the data into Train Set and Test Set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

## Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 8)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


# Applying Kernel PCA
#from sklearn.decomposition import KernelPCA
#kpca = KernelPCA(n_components = 6, kernel = 'rbf')
#X_train = kpca.fit_transform(X_train)
#X_test = kpca.transform(X_test)

# --------------------------------Fitting the model to the Training set
#from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import PolynomialFeatures
#poly_reg = PolynomialFeatures(degree = 3)
#X_poly = poly_reg.fit_transform(X_train)
#poly_reg.fit(X_poly, y_train)
#lin_reg_2 = LinearRegression()
#lin_reg_2.fit(X_poly, y_train)
from sklearn.ensemble import GradientBoostingRegressor

gradientBoostingRegressor = GradientBoostingRegressor()
gradientBoostingRegressor.fit(X_train,y_train)

# --------------------------------Predicting the Test set results
#ypred = xgbReg.predict(X_test)
y_pred = gradientBoostingRegressor.predict(X_test)

# --------------------------------- Calculate RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
sqrt(mean_squared_error(y_test, y_pred))
#Or
sqrt(np.mean((y_test-y_pred)**2))











