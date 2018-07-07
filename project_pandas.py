# -*- coding: utf-8 -*-
"""
Created on Sat May 26 13:51:55 2018

@author: Shariq
"""

import random
import pandas as pd
import numpy as np
import re
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression


"""a = np.array([10,20,30,40,50])
aString = "np.sqrt(a)"
eval(aString)"""

test = pd.read_csv("test.csv")
test.iloc[0,0]

"""Normalizing the dataset using preprocessing"""
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(test)
test = pd.DataFrame(x_scaled)

test.columns = ['X1','X2','X3','X4','X5','y']


X = test.iloc[:,:5]
y = test.iloc[:,5]
funList = ["np.log", "*", "np.exp", "np.sqrt"]

popsize = 15

def randFeature():
    ranFun = random.choice(funList)
    if ranFun == "*":
        return ''.join([random.choice(X.columns),ranFun,random.choice(X.columns)])
    else:
        return ''.join([ranFun,'(',str(random.choice(X.columns)),')'])

"""Transformation"""
def init():
    gen1 = np.array([])
    while gen1.size < 15:
        n=random.randint(1,X.shape[1])
        setOfInd = set() 
        while len(setOfInd) < n:
            feature = randFeature()
            setOfInd.add(feature)
        gen1 = np.append(gen1,setOfInd)
    return gen1
        
        
gen = init()
n = len(gen) 
gen = np.reshape(gen,(n,1))
gen.shape

"""Evaluate"""
def updatedEvalString(s):
    """Indentifying the features and filtering out the unique features"""
    featureNum = np.unique(re.findall('\d+', s))
    for m in featureNum:
        s = s.replace(str.format("X{0}",m),str.format("X['X{0}']",m))
    return s

"""Run ML algorithm to find the accuracy of each individual(Cross validation)"""
"""regressor = LinearRegression()
regressor.fit(indMatrix, y)
yhat = regressor.predict(indMatrix)

linRMSE = mean_squared_error(y, yhat)
linRMSE"""

def score(inEval):
    length = len(inEval)
    indMatrix = pd.DataFrame()
    i=0
    listEval = list(inEval)
    for ele in listEval:
        evalString = updatedEvalString(ele)
        #Exception handling against log(0)
        try:
            indMatrix[str.format('col{0}',i)] = eval(evalString)    
        except ZeroDivisionError:
            continue     
        i = i+1
    """Remove inf with 1 """
    indMatrix = indMatrix.replace([np.inf, -np.inf], 1)
    regr = ElasticNetCV(cv=5, random_state=0)
    regr.fit(indMatrix,y)
    return (regr.score(indMatrix,y))

#print(score(gen[1]))

pc= 0.5

def crossover(gen,pc):
    lenGen = len(gen)
    crossArray = gen[np.random.choice(gen.shape[0],int(pc*lenGen), replace = False), :]
    r2 = np.zeros(len(crossArray))
    r2 = np.reshape(r2,(len(r2),1))
    crossArrayR2 = np.append(crossArray, r2, axis=1)
    for genEle in crossArrayR2:
        genEle[1] = score(genEle[0])
    print(crossArrayR2)
    crossArrayR2 = crossArrayR2[crossArrayR2[:,1].argsort()[::-1]]
    newGen = np.setdiff1d(gen, crossArray)
    newGen = np.append(newGen,crossArrayR2[:,0])
    newGen = np.reshape(newGen,(len(newGen),1))
    return newGen

pm = 0.5
def mutation(gen, pm):
    lenGen = len(gen)
    mutArray = gen[np.random.choice(gen.shape[0],int(pm*lenGen), replace = False), :]
    for genEle in mutArray:
        n = np.random.choice(2)
        if (n%2 == 0 or len(genEle[0])==1):
            feature = randFeature()
            genEle[0].add(feature)
        else:
            genEle[0].pop()
    newGen = np.setdiff1d(gen, mutArray)
    newGen = np.append(newGen,mutArray)
    return newGen

