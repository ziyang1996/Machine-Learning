''' Task 4 test

    Test data should be saved as 'test set.txt' file in the same folder.
    The model parameters should be read from 'problem4_parameters.pkl' file
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

def getScore(A):
    Model=[]
    for i in range(3):
        clf=joblib.load('model/problem4_parameters-model-'+str(A)+'-'+str(i+1)+'.pkl')
        Model.append(clf)

    predict=Predict(Model,test_X)
    score=Accuracy(predict,test_Y)
    print('')
    print('ANN structure: [ 256, '+str(A)+' , 1 ] ')
    print('prediction accuracy = ',round(score*100,6),'%')
    print('')
    return predict,score

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

#--------------------------------main function--------------------------
if __name__ =="__main__":
    f1='data/test_set.txt'
    test_X,test_Y=load_data(f1)   
	
    print('test data prediction accuracy :')

    A=(30,10)
    getScore(A)

    A=(60,20)
    getScore(A)
    
    A=(60,30,10)
    getScore(A)
    
    A=(120,40,10)
    predict,score=getScore(A)
    print('')
    print('------------------------------------------------------------------')    
    print('')
    print('use ANN model with [ 256, '+str(A)+' , 1 ] stucture to recognition digits:')

    print('')
    m = min(test_X.shape[0],10)
    print('Randomly select',m,'data to display and compare the recognition results')
    print('original digit: ')
    for i in range(m):
        print('\t',int(test_Y[i,0]),end='')
    print('')
    print('classification results:')
    for i in range(m):
        print('\t',int(predict[i,0]),end='')
    print('')
    print('')

    os.system('pause')
