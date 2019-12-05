''' Task 2 train
    Apply neural network of 1 hidden layer to classify 1 and 5. 
    The features are: symmetry and average intensity.
    Here are 4 units in the hidden layer. 
    The train data should be saved as 'train features.txt' file in the same folder
    this program will generate a 'problem2_parameters.pkl' to save model parameters
'''
from numpy import *
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
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
    ax.set_title('train data examine( digit:'+str(A)+' )')
    plt.show()



#--------------------------------train----------------------------------

def train(train_X, train_Y, unit):
    Model = []

    kf = KFold(n_splits=3)
    bestScore = 0.0
    i=1
    best=0
    print('ANN Model: single hidden layer with',unit,'units:')
    print('')
    print('we use 3-fold crose-validation to get three model:')
    for train,test in kf.split(train_X):
        clf = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=unit, random_state=1) 
        clf.fit(train_X[train],train_Y[train])
        score = clf.score(train_X[test],train_Y[test])
        print('fold',i,'score : ',score)
        i+=1
        Model.append(clf)
    
    return Model


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
    f1='data/train_features.txt'
    data_X,data_Y=load_data(f1)   
    # 提取出所需的数字
    train_X,train_Y=get_number(data_X,data_Y,[1,5])
    #print(train_X.shape)
    #print(train_Y.shape)
    #plot(train_X,train_Y,[1,5])
    bestModel = []
    bestScore = 0.0
    unit = 0
    for i in range(5,6):
        model=train(train_X,train_Y,i)
        predict=Predict(model,train_X)
        score=Accuracy(predict,train_Y)
        print('')
        print('train data accuracy with',i,'units :',round(score*100,5),'%')
        print('')
        if score>bestScore:
            bestModel = model
            bestScore = score
            unit = i
        print('--------------------------------------------------------')
    # save model
    print('')
    for i in range(3):
        joblib.dump(bestModel[i],'model/problem2_parameters-model'+str(i+1)+'.pkl')
    print('model saved successfully!')
    #clf=joblib.load('1.pkl')


    print('')
    print('whole train data accuracy = ',round(bestScore*100,5),'%')
    print('whole train data error = ',round((1.0-bestScore),5))
    print('')
    print('sturcture of the hidden layer: ')
    print('(',unit,'units in 1 hidden layer)')

    predict = Predict(bestModel,train_X)
    #print(train_X.shape,predict.shape)
    plot(train_X,predict,[1,5])
   
    print('')
    os.system('pause')
    