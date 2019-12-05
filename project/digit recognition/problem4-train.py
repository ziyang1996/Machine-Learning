''' Task 3 train

    Train data should be saved as 'train set.txt' file in the same folder.
    this program will generate a 'problem4_parameters.pkl' to save model parameters.
'''

from numpy import *
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.externals import joblib
import os
import warnings
warnings.filterwarnings('ignore')


#-----------------------------load data------------------------------

def load_data(file):
    f=open(file,"r")
    X=[]
    Y=[]
    for l in f.readlines():
        l=l.strip().split(' ')
        _X=[]
        Y.append(float(l[0]))
        for i in range(1,len(l)):
            if l[i]!='':
                _X.append(float(l[i]))
        X.append(_X)

    return mat(X),mat(Y).T


#--------------------------------predict----------------------------------

def Predict(Model,X):
    y1=mat(Model[0].predict(X)).T
    y2=mat(Model[1].predict(X)).T
    y3=mat(Model[2].predict(X)).T
    Y=[]
    m = X.shape[0]
    for i in range(m):
        if y1[i,0]==y2[i,0] or y1[i,0]==y3[i,0]:
            Y.append(y1[i,0])
        else:
            Y.append(y2[i,0])
    return mat(Y).T

#--------------------------------accuracy----------------------------------

def Accuracy(Y,_Y):
    right=0
    m=Y.shape[0]
    for i in range(m):
        if Y[i,0]==_Y[i,0]:
            right+=1
    return right/m

#--------------------------------train----------------------------------

def train(train_X, train_Y, unit):
    Model = []
    kf = KFold(n_splits=3)
    
    i=1
    print('ANN Model: hidden layer with',unit,'structure:')
    print('')
    print('we use 3-fold crose-validation to get three models:')
    error=0.0
    for train,test in kf.split(train_X):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=unit, random_state=1) 
        clf.fit(train_X[train],train_Y[train])
        score = clf.score(train_X[test],train_Y[test])
        print('fold',i,'score : ',score)
        error += 1-score
        Model.append(clf)
        i+=1
    
    predict=Predict(Model,train_X)
    score=Accuracy(predict,train_Y)
    
    print('3-fold cross-validation mean error is:',round(error/3.0,6))
    print('')
    print('train data accuracy with hidden layer ',unit,': ',round(score*100,6),'%')
    print('')
    print('----------------------------------------------------------------------')
    for i in range(3):
        joblib.dump(Model[i],'model/problem4_parameters-model-'+str(A)+'-'+str(i+1)+'.pkl')
    return Model,score

#--------------------------------main function--------------------------
if __name__ =="__main__":
    f1='data/train_set.txt'
    train_X,train_Y=load_data(f1)   

    bestModel = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=6, random_state=1) 
    bestScore = 0.0
    unit = 0

    A=(30,10)
    Model,score=train(train_X,train_Y,A)
    if score>bestScore:
        bestModel = Model
        bestScore = score
        unit = A
    
    A=(60,20)
    Model,score=train(train_X,train_Y,A)
    if score>bestScore:
        bestModel = Model
        bestScore = score
        unit = A

    A=(60,30,10)
    Model,score=train(train_X,train_Y,A)
    if score>bestScore:
        bestModel = Model
        bestScore = score
        unit = A

    A=(120,40,10)
    Model,score=train(train_X,train_Y,A)
    if score>bestScore:
        bestModel = Model
        bestScore = score
        unit = A
    
    # save model
    print('')
    print('model saved successfully!')
    #clf=joblib.load('1.pkl')

    print('')
    print('final model structure: [ 256,',unit,', 1 ]')
    print('')
    print('whole train data accuracy = ',round(bestScore*100,5),'%')
    print('whole train data error = ',round((1.0-bestScore),5))
    
    os.system('pause')