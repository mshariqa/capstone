# -*- coding: utf-8 -*-
"""
Created on Sat May 26 13:51:55 2018

@author: Isha and Shariq
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

def replaceZeroes(data):
    data[data == 0] = 10**-4
    return data

test = pd.read_csv("test.csv")
test.iloc[0,0]

"""Normalizing the dataset using preprocessing"""
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(test)
#Replace all 0 with a minimum value close to zero to resolve log(0) issue
x_scaled = replaceZeroes(x_scaled)
test = pd.DataFrame(x_scaled)

test.columns = ['X1','X2','X3','X4','X5','y']


X = test.iloc[:,:5]
y = test.iloc[:,5]

funList = ["np.log", "*", "np.exp", "np.sqrt"]
setOpList = ["union","sym_diff"]

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
    while len(gen1) < 15:
        n=random.randint(1,X.shape[1])
        setOfInd = set()
        while len(setOfInd) < n:
            feature = randFeature()
            setOfInd.add(feature)
        if(setOfInd not in gen1):
            gen1 = np.append(gen1,setOfInd)
    gen1 = np.reshape(gen1,(len(gen1),1))    
    return gen1
        
"""Evaluate"""
def updatedEvalString(s):
    """Indentifying the features and filtering out the unique features"""
    featureNum = np.unique(re.findall('\d+', s))
    for m in featureNum:
        s = s.replace(str.format("X{0}",m),str.format("X['X{0}']",m))
    return s

def score(inEval):
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
    regr = ElasticNetCV(cv=5, random_state=0, max_iter=5000)
    regr.fit(indMatrix,y)
    return (regr.score(indMatrix,y))

pc= 0.5
def getCrossover(crossEle1, crossEle2):
    ranOp = random.choice(setOpList)
    if ranOp == "union":
        return crossEle1.union(crossEle2)
    elif ranOp == "intersection":
        return np.intersect1d(crossEle1, crossEle2)
    elif ranOp == "sym_diff":
        return crossEle1.symmetric_difference(crossEle2)
    
def crossover(gen,pc):
    gen = init()
    lenGen = len(gen)
    numCross = int(pc*lenGen)
    i = 0
    crossGen = np.array([]) #crossovered population
    r2 = np.zeros(len(gen))
    r2 = np.reshape(r2,(len(r2),1))
    
    # Added R^2 element in gen
    gen = np.append(gen, r2, axis=1)
    for genEle in gen:
        genEle[1] = score(genEle[0])
    #Calulation score for the randomly selected individual from population
    while i<numCross:
        #Selecting random values from the population for crossover
        crossArray = gen[np.random.choice(gen.shape[0],numCross , replace = False), :]
        crossArray = crossArray[crossArray[:,1].argsort()[::-1]]
        if numCross == 0:
            break
        elif numCross == 1:
            crossGen = np.append(crossGen, crossArray[0,0])
        else:
            crossGen = np.append(crossGen,getCrossover(crossArray[0,0], crossArray[1,0]))
        i = i+1        
    gen = gen[gen[:,1].argsort()[::-1]]
    # Selection and crossover to the new generation 
    newGen = gen[:,0]
    #print("numCross", numCross, "lenGen", lenGen)
    newGen[lenGen - len(crossGen):] = crossGen
    newGen = np.reshape(newGen,(len(newGen),1))
    return newGen

pm = 0.5
def mutation(gen, pm):
    gen = init()
    lenGen = len(gen)
    mutArray = gen[np.random.choice(gen.shape[0],int(pm*lenGen), replace = False), :]
    for genEle in mutArray:
        n = np.random.choice(2)
        if (n%2 == 0 or len(genEle[0])==1):
            genEle[0].add(randFeature())
        else:
            genEle[0].pop()
    newGen = np.setdiff1d(gen, mutArray)
    newGen = np.append(newGen,mutArray)
    newGen = np.reshape(newGen,(len(newGen),1))
    return newGen

def main():
    iterations = 10
    i = 0 
    newGen = init()
    indbest = set()
    fbest = 0 
    while i < iterations:
        r2 = np.zeros(len(newGen))
        r2 = np.reshape(r2,(len(r2),1))
    
        # Added R^2 element in gen
        newGen = np.append(newGen, r2, axis=1)
        for genEle in newGen:
            genEle[1] = score(genEle[0])
    
        newGen = newGen[newGen[:,1].argsort()[::-1]]
        if fbest < newGen[0,1]:
            indbest = newGen[0,0]
            fbest = newGen[0,1]
        print(i)
        newGen = crossover(newGen,pc)
        newGen = mutation(newGen,pm)
        i = i+1
    print(indbest)
    print(fbest)
    
main()