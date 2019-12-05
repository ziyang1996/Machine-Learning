''' Task 3 train
	Apply neural network of 2 hidden layer to classify 1 and 5,
	using the raw features as input.
    there are 6 units in the 1st hidden layer and 2 units in the 2nd hidden layer.
    Train data should be saved as 'train set.txt' file in the same folder.
    this program will generate a 'problem3_parameters.pkl' to save model parameters.
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


def get_number(data_X,data_Y,A):
    m,n = data_X.shape
    X=[]
    Y=[]
    for i in range(m):
        if int(data_Y[i,0]) in A:
            _X=[]
            for j in range(n):
                _X.append(data_X[i,j])
            X.append(_X)
            Y.append(data_Y[i,0])
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

#--------------------------------main function--------------------------
if __name__ =="__main__":
    f1='data/train_set.txt'
    data_X,data_Y=load_data(f1)   
    # 提取出所需的数字
    train_X,train_Y=get_number(data_X,data_Y,[1,5])
    #print(train_X.shape)
    #print(train_Y.shape)
	
    Model1 = []
    kf = KFold(n_splits=3)
    i=1

    print('ANN Model: 2 hidden layer with 6 units and 2 units respectively:')
    print('')
    print('we use 3-fold crose-validation to get three model:')
    
    for train,test in kf.split(train_X):
        clf = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(6,2), random_state=1) 
        clf.fit(train_X[train],train_Y[train])
        score = clf.score(train_X[test],train_Y[test])
        print('model',i,'score : ',score)
        Model1.append(clf)
        i+=1

    for i in range(3):
        joblib.dump(Model1[i],'model/problem3_parameters-model-1-'+str(i+1)+'.pkl')
    print('model saved successfully!')

    predict = Predict(Model1,train_X)
    score = Accuracy(predict,train_Y)
    print('')
    print('ANN Model: 2 hidden layers with 6 units and 2 units:')
    print('whole train data examining accuracy = ',round(score*100,6),'%')
    print('')

    print('------------------------------------------------------------------')


    Model1 = []
    kf = KFold(n_splits=3)
    i=1

    print('ANN Model: 2 hidden layer with 3 units and 2 units respectively:')
    print('')
    print('we use 3-fold crose-validation to get three model:')
    
    for train,test in kf.split(train_X):
        clf = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(3,2), random_state=1) 
        clf.fit(train_X[train],train_Y[train])
        score = clf.score(train_X[test],train_Y[test])
        print('model',i,'score : ',score)
        Model1.append(clf)
        i+=1

    for i in range(3):
        joblib.dump(Model1[i],'model/problem3_parameters-model-2-'+str(i+1)+'.pkl')
    print('model saved successfully!')

    predict = Predict(Model1,train_X)
    score = Accuracy(predict,train_Y)
    print('')
    print('ANN Model: 2 hidden layers with 3 units and 2 units:')
    print('whole train data examining accuracy = ',round(score*100,6),'%')
    print('')

    print('------------------------------------------------------------------')


    os.system('pause')

