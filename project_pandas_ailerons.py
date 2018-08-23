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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from operator import itemgetter
import warnings
import winsound
import time
from sklearn.metrics import r2_score

frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second

warnings.filterwarnings("ignore")

# Crossover probability of set operations
PROB_UNION = 0.3
PROB_INTERSECTION = 0.1
PROB_SYMDIFF = 0.3
PROB_ELEMENTCROSSOVER = 0.3

# Mutation probability for ADD and REMOVE
PROB_ADD = 0.6
PROB_REMOVE = 0.4

def replaceZeroes(data):
    data[data == 0] = 10**-4
    return data

#test = pd.read_csv("boston_house_data.csv")
#test = pd.read_csv("ailerons.csv")
#test = pd.read_csv("forestFires.csv")


#### Only run for abalone dataset
test = pd.read_csv("abalone.csv")
test = pd.get_dummies(test)
cols = test.columns.tolist()
cols = cols[-3:] + cols[:-3]
test = test[cols]
####

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

# regularization variable for individual's size
max_len = int(X.shape[1]*15)

# List of non-linear functions for feature transformation
funList = ["np.log", 
           "*", 
           "np.exp", 
           "np.sqrt", 
           "np.square", 
           "np.log10"]

# set operations to be performed for crossover
setOpList = ["union", 
             "sym_diff", 
             "intersection",
             "elementCrossover"
             ]



# Function to create random non linear features 
def randFeature():
    ranFun = random.choice(funList)
    #ranFun = np.random.choice(funList, p = [0.2,0.2,0.2,0.2,0,0.2])
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

# Evaluates the individuals
def evaluatedMatrix(inEval, X):
    indMatrix = pd.DataFrame()
    i=0
    for ele in inEval:
        evalString = updatedEvalString(ele)
        #Exception handling against log(0)
        try:
            indMatrix[str.format('col{0}',i)] = eval(evalString)    
        except ZeroDivisionError:
            continue     
        i = i+1
    # Remove inf with 1
    indMatrix = indMatrix.replace([np.inf], 1)
    indMatrix = indMatrix.replace([-np.inf], 0.0001)
    return indMatrix

# Calculate R^2 value for an individual 
def score(inEval, X, y):
    indMatrix = pd.DataFrame()
    
    listEval = list(inEval)
    indMatrix = evaluatedMatrix(listEval, X)
    
    # Linear regression with elastic net
    #regr = ElasticNet(random_state=0, l1_ratio=0, alpha = 1)
    regr = ElasticNetCV(random_state=0)
    regr.fit(indMatrix,y)
    return (regr.score(indMatrix,y))

# Select an element from the one parent and add it to other parent
def elementCrossover(crossEle1, crossEle2):
    n=random.randint(0,1)
    try:
        if n == 0:
            crossEle1.union(set(random.sample(crossEle2, 1)))
            return crossEle1
        else:
            crossEle2.union(set(random.sample(crossEle1, 1)))
            return crossEle2
    except KeyError as err:
        print('Handling run-time error:', err)


# Set crossover function
def getCrossover(crossEle1, crossEle2):
    ranOp = np.random.choice(setOpList, 1, p=[PROB_UNION,  PROB_SYMDIFF, PROB_INTERSECTION, PROB_ELEMENTCROSSOVER])
    if "union" in ranOp:
        return crossEle1.union(crossEle2)
    elif "intersection" in ranOp:
        return np.intersect1d(crossEle1, crossEle2)
    elif "sym_diff" in ranOp:
        return crossEle1.symmetric_difference(crossEle2)
    elif "elementCrossover" in ranOp:
        return elementCrossover(crossEle1, crossEle2)

# Sort method
def sortby(gen, n):
    nlist = [(x[n], x) for x in gen]
    try:
        nlist = sorted(nlist, key=itemgetter(0), reverse = True)
    except ValueError:
        print("Error nlist:", nlist)
    return [val for (key, val) in nlist]

# Crossover for next generation
def crossover(gen,pc):
    lenGen = len(gen)
    numCross = int(pc*lenGen)
    i = 0
    crossGen = np.array([]) #crossovered population
    crossArray = np.array([])

    while i<numCross:
        #Selecting random values from the population for crossover
        crossArray = gen[np.random.choice(gen.shape[0],numCross , replace = False), :]
        #crossArray = crossArray[crossArray[:,1].argsort()[::-1]]
        # sort crossArray in descending order
        crossArray = sortby(crossArray, 1)
        if numCross == 0 or numCross == 1:
            break
        else:
            crossEle = getCrossover(crossArray[0][0], crossArray[1][0])
            if(not(np.isin(crossEle,gen[:,0])) and crossEle not in crossGen and len(crossEle)<max_len and len(crossEle)>0):
                crossGen = np.append(crossGen,crossEle)
                i = i+1       
    gen = sortby(gen, 1)
    # Selection and crossover to the new generation 
    newGen = [a[0] for a in gen]
    newGen[lenGen - len(crossGen):] = crossGen
    newGen = np.reshape(newGen,(len(newGen),1))
    return newGen

# Mutation for next generation  
def mutation(gen, pm):
    lenGen = len(gen)
    mutArray2 = copy.deepcopy(gen[np.random.choice(gen.shape[0],int(pm*lenGen), replace = False), :])
    mutArray = copy.deepcopy(mutArray2)
    for genEle in mutArray:
        n = np.random.choice(2,p=[PROB_ADD,  PROB_REMOVE])
        if ((n%2 == 0 or len(genEle[0])==1) and len(genEle[0])<max_len):
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

def coefStr(x):
    """Gracefully print a number to 3 significant digits.  See _testCoefStr in
    unit tests"""
    if x == 0.0:
        s = '0'
    elif np.abs(x) < 1e-4: s = ('%.2e' % x).replace('e-0', 'e-')
    elif np.abs(x) < 1e-3: s = '%.6f' % x
    elif np.abs(x) < 1e-2: s = '%.5f' % x
    elif np.abs(x) < 1e-1: s = '%.4f' % x
    elif np.abs(x) < 1e0:  s = '%.3f' % x
    elif np.abs(x) < 1e1:  s = '%.2f' % x
    elif np.abs(x) < 1e2:  s = '%.1f' % x
    elif np.abs(x) < 1e4:  s = '%.0f' % x
    else:                     s = ('%.2e' % x).replace('e+0', 'e')
    return s

# Calculates test accuracy for an individual 
def calculateAccuracy(indbest, X_train, y_train, X_test, y_test):
    indbest = list(indbest) 
    evalTrain = evaluatedMatrix(indbest, X_train)
    evalTest = evaluatedMatrix(indbest, X_test)
    
    # Linear regression with elastic net
    regr = ElasticNetCV(random_state=0)
    regr.fit(evalTrain,y_train)
    y_pred = regr.predict(evalTest)
    print("Test Accuracy: ", r2_score(y_test, y_pred))
    return r2_score(y_test, y_pred)

# Sort method
def sortCoef(indbest, coef):
    nlist = [(y, x) for x,y in zip(indbest, coef)]
    try:
        nlist = sorted(nlist, key=itemgetter(0), reverse = True)
    except ValueError:
        print("Error nlist:", nlist)
    return [val for (key, val) in nlist], [key for (key, val) in nlist]

# Calculates test accuracy for an individual and print the model cost function
def calculateAccuracyWithModel(indbest, X_train, y_train, X_test, y_test):
    indbest = list(indbest) 
    evalTrain = evaluatedMatrix(indbest, X_train)
    evalTest = evaluatedMatrix(indbest, X_test)
    
    # Linear regression with elastic net
    regr = ElasticNetCV(random_state=0)
    regr.fit(evalTrain,y_train)
    y_pred = regr.predict(evalTest)
    print(r2_score(y_test, y_pred))
    indbest, regr.coef_ = sortCoef(indbest, regr.coef_)
    model = ""
    i=0
    if regr.intercept_ not in [0,-0]:
        model = str(coefStr(regr.intercept_))
    for ind in indbest:
        if regr.coef_[i] not in [0,-0]: 
            if "-" in str(regr.coef_[i]): 
                indCoef = str(coefStr(regr.coef_[i]))+"*"+str(ind) 
            elif len(model) > 0:   
                indCoef = "+" + str(coefStr(regr.coef_[i]))+"*"+ ind
            else:
                indCoef = str(coefStr(regr.coef_[i]))+"*"+ ind
            model = model + indCoef
        i = i + 1
    print(model)


# Main function
def geneticAlgorithm():
    # Number of iterations of genetic algorithm 
    generation = 25
    i = 1
    # Crossover probability:
    # Used for calculating the population percentage for crossover 
    pc = 0.7
    # Mutation probability:
    # Used for calculating the population percentage for mutation 
    pm = 0.7
    # Initial population
    newGen = init()
    indbest = set()
    fbest = 0 
    # Intilize train and test accuaracy array
    trainAccuracy = np.array([])
    testAccuracy = np.array([])
    
    while i <= generation:
        # To add R^2 column in gen
        r2 = np.zeros((len(newGen),1))
        newGen = np.append(newGen, r2, axis=1)
        # Calculates the R^2 score corresponding to each individual 
        for genEle in newGen:
            genEle[1] = score(genEle[0],X_train, y_train)
        
        # Sort individual as per their r^2 values 
        newGen = newGen[newGen[:,1].argsort()[::-1]]
        
        # Find the best individual in the generation and compare it with best individual so far
        if fbest < newGen[0,1]:
            indbest = newGen[0,0]
            fbest = newGen[0,1]
        
        trainAccuracy = np.append(trainAccuracy,fbest)
        testAccuracy = np.append(testAccuracy,calculateAccuracy(indbest, X_train, y_train, X_test, y_test))
        
        print("iterations:",i)
        #print("Best Individual:",indbest)
        #print("Best fitness:",fbest)
        
        # Crossover for next generation
        newGen = crossover(newGen,pc)
        
        # Mutation for next generation
        newGen = mutation(newGen,pm)
        i = i+1
    #print("Final best Individual:",indbest)
    #print("Final best fitness:",fbest)
    calculateAccuracyWithModel(indbest, X_train, y_train, X_test, y_test)
    # Data
    """df=pd.DataFrame({'x': range(1,16), 'y1': trainAccuracy, 'y2':  testAccuracy})
 
    # multiple line plot
    plt.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4, label="trainAccuracy")
    plt.plot( 'x', 'y2', data=df, marker='', color='olive',     linewidth=2, label="testAccuracy")
    plt.legend()"""
    return indbest, fbest

start_time = time.time()

res = pd.DataFrame(columns=('ind', 'R2', 'testAccuracy'))
for i in range(5):
    indbest, fbest = geneticAlgorithm()
    testAccuracy = calculateAccuracy(indbest, X_train, y_train, X_test, y_test)
    res.loc[i] = [indbest,fbest,testAccuracy]

print("--- %s seconds ---" % (time.time() - start_time))

res
res.to_csv("abalone_result.csv", encoding='utf-8', index=True)

# Make a beep sound when finishes
winsound.Beep(frequency, duration)


"""
# if want to run only one time
indbest, fbest = geneticAlgorithm()

# Elastic net without GA
regr = ElasticNetCV(random_state=0)
regr.fit(X_train,y_train)
regr.score(X_train,y_train)

regr.score(X_test, y_test)

# Linear Regression without GA
lm = LinearRegression()
lm.fit(X_train, y_train)

scores = cross_val_score(lm, X, y, scoring="r2", cv=5)
np.average(scores)

lm.score(X_test, y_test)


## FFX
import ffx
models = ffx.run(X_train, y_train, X_test, y_test, FFXColsName)

X_test_matrix = X_test.as_matrix()

for model in models:
    yhat = model.simulate(X_test_matrix)
    print(r2_score(y_test, yhat))
    print(model)
    
"""
