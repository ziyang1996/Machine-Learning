''' Task 2 test
    Apply neural network of 1 hidden layer to classify 1 and 5. 
    The features are: symmetry and average intensity.
    There are 4 units in the hidden layer.
    The model parameters should be read from 'problem2_parameters.pkl' file
    Test data should be saved as 'test features.txt' file in the same folder
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


#-----------------------------show plot------------------------------


def plot(X,Y,A):
    m,n = X.shape
    x1 = [[] for i in range(10)]
    x2 = [[] for i in range(10)]
    for i in range(m):
        a = int(Y[i,0])
        if a in A:
            x1[a].append(float(X[i,0]))
            x2[a].append(float(X[i,1]))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(10):
        ax.scatter(x1[i],x2[i],s=3)
    ax.set_xlabel('Intensity')
    ax.set_xlabel('Symmetry')
    ax.set_title('test data prediction( digit:'+str(A)+' )')
    plt.show()


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
    f1='data/test_features.txt'
    data_X,data_Y=load_data(f1)   
    # 提取出所需的数字
    test_X,test_Y=get_number(data_X,data_Y,[1,5])

    unit=4
    # read parameters
    model=[]
    for i in range(3):
        clf=joblib.load('model/problem2_parameters-model'+str(i+1)+'.pkl')
        model.append(clf)

    predict = Predict(model,test_X)
    score = Accuracy(predict,test_Y)
    print('test data prediction accuracy =',round(score*100,6),'%')
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
    plot(test_X, predict, [1,5])

    os.system('pause')



