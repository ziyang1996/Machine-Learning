''' Task 3 train-2
	
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

def iteration(Max,train_X,train_Y,test_X,test_Y):
    bestModel = MLPClassifier(solver='sgd', max_iter=Max, hidden_layer_sizes=(6,3), random_state=1) 

    kf = KFold(n_splits=3)
    bestScore = 0.0
    i=1
    best=0
    for train,test in kf.split(train_X):
        clf = MLPClassifier(solver='sgd', max_iter=Max, hidden_layer_sizes=(6,3), random_state=1) 
        clf.fit(train_X[train],train_Y[train])
        score = clf.score(train_X[test],train_Y[test])
        if score>bestScore:
            bestModel=clf
            bestScore=score
            best=i
        i+=1

    return 1.0-bestModel.score(train_X,train_Y),1.0-bestModel.score(test_X,test_Y)

#--------------------------------main function--------------------------
if __name__ =="__main__":
    f1='data/train_set.txt'
    f2='data/test_set.txt'
    data_X,data_Y=load_data(f1)   
    test_X,test_Y=load_data(f2)
    # 提取出所需的数字
    train_X,train_Y=get_number(data_X,data_Y,[1,5])
    test_X,test_Y=get_number(test_X,test_Y,[1,5])
    #print(train_X.shape)
    #print(train_Y.shape)
    print('')
    print('Please wait for training. . . . . .')
    print('')
    X=[]
    Y1=[]
    Y2=[]
    for i in range(1,51):
        score1,score2 = iteration(i,train_X,train_Y,test_X,test_Y)
        Y1.append(score1*100)
        Y2.append(score2*100)
        X.append(i)
    plt.figure()
    plt.plot(X,Y1)
    plt.plot(X,Y2)
    plt.xlabel('Iteration')
    plt.ylabel('Error rate ( % )')
    plt.title('in-sample error rate(blue) and test-data error rate(red)')
    plt.show()

