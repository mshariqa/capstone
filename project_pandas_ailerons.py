# -*- coding: utf-8 -*-
"""
Created on Sat May 26 13:51:55 2018

@author: Isha and Shariq
"""

import random
import copy
import pandas as pd
import numpy as np
import re
from sklearn import preprocessing
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

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
XColsName.append('y')
XColsName

test.columns = XColsName

X = test.iloc[:,:-1]
y = test.iloc[:,-1]

# List of non-linear functions for feature transformation
funList = ["np.log", "*", "np.exp", "np.sqrt"]

# set operations to be performed for crossover
setOpList = ["union","sym_diff"]

# Function to create random non linear features 
def randFeature():
    ranFun = random.choice(funList)
    if ranFun == "*":
        return ''.join([random.choice(X.columns),ranFun,random.choice(X.columns)])
    else:
        return ''.join([ranFun,'(',str(random.choice(X.columns)),')'])

# Creates initial population
def init():
    gen1 = np.array([])
    while len(gen1) < 50:
        n=random.randint(1,X.shape[1])
        setOfInd = set()
        while len(setOfInd) < n:
            feature = randFeature()
            setOfInd.add(feature)
        if(setOfInd not in gen1):
            gen1 = np.append(gen1,setOfInd)
    gen1 = np.reshape(gen1,(len(gen1),1))    
    return gen1
        
# Converts the individual set element string to feature of X
# eg. 'np.exp(X2)' is converted to 'np.exp(X['X2'])'
def updatedEvalString(s):
    """Indentifying the features and filtering out the unique features"""
    if (s.find("*") == -1):
        featureNum = np.unique(re.findall('\d+', s))
        for m in featureNum:
            s = s.replace(str.format("X{0}",m),str.format("X['X{0}']",m))
    else:
        pos = s.find("*")
        s = str.format("X['{0}']*X['{1}']",s[0:pos],s[pos+1:])
    return s

# Calculate R^2 value for an individual 
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
    # Remove inf with 1
    indMatrix = indMatrix.replace([np.inf, -np.inf], 1)
    # Linear regression with elastic net
    regr = ElasticNetCV(cv=5, random_state=0, max_iter=5000)
    regr.fit(indMatrix,y)
    
    return (regr.score(indMatrix,y))

def linearRegressionScore(inEval):
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
    # Remove inf with 1
    indMatrix = indMatrix.replace([np.inf, -np.inf], 1)
    
    # Linear regression
    lm = LinearRegression()
    lm.fit(X, y)

    scores = cross_val_score(lm, X, y, scoring="r2", cv=5)
    return (np.average(scores))

    
# Set crossover function
def getCrossover(crossEle1, crossEle2):
    ranOp = random.choice(setOpList)
    if ranOp == "union":
        return crossEle1.union(crossEle2)
    elif ranOp == "intersection":
        return np.intersect1d(crossEle1, crossEle2)
    elif ranOp == "sym_diff":
        return crossEle1.symmetric_difference(crossEle2)
   
# Crossover for next generation
def crossover(gen,pc):
    # The function still returns dublicates 
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
        if numCross == 0 or numCross == 1:
            break
        else:
            crossEle = getCrossover(crossArray[0,0], crossArray[1,0])
            if(crossEle not in gen and crossEle not in crossGen):
                crossGen = np.append(crossGen,crossEle)
                i = i+1       
    gen = gen[gen[:,1].argsort()[::-1]]
    # Selection and crossover to the new generation 
    newGen = gen[:,0]
    #print("numCross", numCross, "lenGen", lenGen)
    newGen[lenGen - len(crossGen):] = crossGen
    newGen = np.reshape(newGen,(len(newGen),1))
    #print("NewGen",newGen)
    #print("Cross Lenght",len(newGen))
    return newGen


# Mutation for next generation  
def mutation(gen, pm):
    lenGen = len(gen)
    mutArray2 = copy.deepcopy(gen[np.random.choice(gen.shape[0],int(pm*lenGen), replace = False), :])
    mutArray = copy.deepcopy(mutArray2)
    for genEle in mutArray:
        n = np.random.choice(2)
        if (n%2 == 0 or len(genEle[0])==1):
            while True:
                genEle[0].add(randFeature())
                if (len(mutArray) == len(np.unique(mutArray)) and genEle[0] not in gen):
                    break
        else:
            while True:
                genEle[0].pop()
                if (len(mutArray) == len(np.unique(mutArray)) and genEle[0] not in gen):
                    break
                genEle[0].add(randFeature())        
    newGen = np.setdiff1d(gen, mutArray2)
    newGen = np.append(newGen,mutArray)
    newGen = np.reshape(newGen,(len(newGen),1))
    return newGen

# Main function
def geneticAlgorithm():
    # Number of iterations of genetic algorithm 
    generation = 10
    i = 0
    # Crossover probability:
    # Used for calculating the population percentage for crossover 
    pc = 0.4
    # Mutation probability:
    # Used for calculating the population percentage for mutation 
    pm = 0.1
    # Initial population
    newGen = init()
    print(newGen)
    indbest = set()
    fbest = 0 
    while i < generation:
        # To add R^2 column in gen
        r2 = np.zeros(len(newGen))
        r2 = np.reshape(r2,(len(r2),1))
        newGen = np.append(newGen, r2, axis=1)
        # Calculates the R^2 score corresponding to each individual 
        for genEle in newGen:
            genEle[1] = score(genEle[0])
        
        # Sort individual as per their r^2 values 
        newGen = newGen[newGen[:,1].argsort()[::-1]]
        
        # Find the best individual in the generation and compare it with best individual so far
        if fbest < newGen[0,1]:
            indbest = newGen[0,0]
            fbest = newGen[0,1]
        
        print("iterations:",i)
        print("Best Individual:",indbest)
        print("Best fitness:",fbest)
        
        # Crossover for next generation
        newGen = crossover(newGen,pc)
        
        # Mutation for next generation
        newGen = mutation(newGen,pm)
        i = i+1
    print("Final best Individual:",indbest)
    print("Final best fitness:",fbest)
    return indbest, fbest
    
res = pd.DataFrame(columns=('ind', 'R2'))
for i in range(10):
    indbest, fbest = geneticAlgorithm()
    res.loc[i] = [indbest,fbest]

res
res.to_csv("ailerons_result.csv", encoding='utf-8', index=True)

# Elastic net without GA
regr = ElasticNetCV(cv=5, random_state=0, max_iter=2000)
regr.fit(X,y)
regr.score(X,y)

# Linear Regression without GA
lm = LinearRegression()
lm.fit(X, y)

scores = cross_val_score(lm, X, y, scoring="r2", cv=5)
np.average(scores)

#  If we take the mean of the training data as a constant prediction for y
from sklearn.metrics import r2_score
y_true = y
y_pred = y.copy()
y_pred[:] = np.mean(y)
r2_score(y_true, y_pred)
