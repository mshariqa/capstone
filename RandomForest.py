# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 23:05:18 2018

@author: Shariq
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import warnings
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

def replaceZeroes(data):
    data[data == 0] = 10**-4
    return data

#test = pd.read_csv("boston_house_data.csv")
#test = pd.read_csv("ailerons.csv")
#test = pd.read_csv("forestFires.csv")

test = pd.read_csv("abalone.csv")
test = pd.get_dummies(test)
cols = test.columns.tolist()
cols = cols[-3:] + cols[:-3]
test = test[cols]

#test = pd.read_csv("airfoil_self_noise.txt", sep = "\t", header = None)

#test = pd.read_csv("Concrete_Data.csv")

test.shape

#Normalizing the dataset ussing preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(test)

#Replace all 0 with a minimum value close to zero to resolve log(0) issue
x_scaled = replaceZeroes(x_scaled)
test = pd.DataFrame(x_scaled)

# Renaming the dataset columns 
# test.columns = ['X1','X2','X3','X4','X5','y']
XColsSize = test.shape[1] - 1
XColsName = ['X{}'.format(x+1) for x in range(0, XColsSize)]
FFXColsName = np.copy(XColsName)
XColsName.append('y')
XColsName

test.columns = XColsName

X = test.iloc[:,:-1]
y = test.iloc[:,-1]

# create training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

regr = RandomForestRegressor(random_state=0, n_estimators=1000)
regr.fit(X_train, y_train)


print(regr.score(X_train,y_train))

print(regr.score(X_test, y_test))
