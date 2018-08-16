# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 15:07:26 2018

@author: Shariq
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def replaceZeroes(data):
    data[data == 0] = 10**-4
    return data

test = pd.read_csv("ailerons.csv")
test.shape

test

#Normalizing the dataset ussing preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(test)

#Replace all 0 with a minimum value close to zero to resolve log(0) issue
x_scaled = replaceZeroes(x_scaled)
test = pd.DataFrame(x_scaled)

# Renaming the dataset columns 
#test.columns = ['X1','X2','X3','X4','X5','y']
XColsSize = test.shape[1] - 1
XColsName = ['X{}'.format(x+1) for x in range(0, XColsSize)]
FFXColsName = np.copy(XColsName)
XColsName.append('y')
XColsName

test.columns = XColsName

X = test.iloc[:,:-1]
y = test.iloc[:,-1]

# create training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# R2 and mean square error
from sklearn.metrics import r2_score, mean_squared_error
y_true = y_test
y_pred = y_test.copy()
y_pred[:] = np.mean(y_train)
r2_score(y_true, y_pred)
mean_squared_error(y_true, y_pred)

## FFX
import ffx
models = ffx.run(X_train, y_train, X_test, y_test, FFXColsName)

X_test_matrix = X_test.as_matrix()

X_test_matrix.shape

for model in models:
    yhat = model.simulate(X_test_matrix)
    print(r2_score(y_true, yhat))
    print(model)