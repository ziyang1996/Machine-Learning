from numpy import *
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.externals import joblib

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import os
import warnings
warnings.filterwarnings('ignore')

#--------------------------------load data---------------------------------

def loadData(file):
	f=open(file,"r")
	X=[]
	for l in f.readlines():
		l=l.split(' ')
		_X=[]
		for i in range(len(l)):
			if l[i].isdigit():
				_X.append(int(l[i]))
		X.append(_X)
	return mat(X)


#--------------------------------error statistic--------------------------


def statistic(_Y,Y):
	Error=0
	Sum=0

	m=Y.shape[0]
	for i in range(m):
		if int(_Y[i,0])!=int(Y[i,0]):
			Error+=1
		Sum+=1
	print('------------------------------------')
	print('error statistic: ')
	print('sum digits: ',Sum)
	print('accuracy: ',round((1.0-Error/Sum)*100.0,3),'%')
	print('------------------------------------')


#---------------------------multi model prediction--------------------------
def multiPredict(K,Model,X):
	predict=[]
	for i in range(K):
		pre=Model[i].predict(X)
		predict.append(pre)
	predict=mat(predict).T
	m,n=predict.shape
	#print(predict)
	ans=[]
	for i in range(m):
		num=[0]*10
		for j in range(n):
			num[int(predict[i,j])]+=1
		Max=0
		k=0
		for j in range(10):
			if num[j]>Max:
				Max=num[j]
				k=j
		ans.append(k)
	return mat(ans).T

def Predict(K,Model,X):
	predict=[]
	for i in range(K):
		pre=Model[i].predict(X)
		predict.append(pre)
	predict=mat(predict).T
	return predict

#--------------------------------main function-----------------------------
if __name__ =="__main__":

	fx='data/train_feature.txt'
	fy='data/train_label.txt'
	train_X=loadData(fx)
	train_Y=loadData(fy)

	fx='data/test_feature.txt'
	fy='data/test_label.txt'
	test_X=loadData(fx)
	test_Y=loadData(fy)

	# SVM
	K=3
	Model = []
	kf = KFold(n_splits=K)
	for train,test in kf.split(train_X):
		clf = svm.SVC()
		clf.fit(train_X[train],train_Y[train])
		Model.append(clf)
	for i in range(K):
		joblib.dump(Model[i],'model/parameters-SVM-'+str(i+1)+'.pkl')
	predict=multiPredict(K,Model,test_X)
	statistic(predict,test_Y)
	
	# Logistic Regression
	K=3
	Model = []
	kf = KFold(n_splits=K)
	for train,test in kf.split(train_X):
		clf=LogisticRegression()
		clf.fit(train_X[train],train_Y[train])
		score = clf.score(train_X[test],train_Y[test])
		Model.append(clf)
	for i in range(K):
		joblib.dump(Model[i],'model/parameters-LR-'+str(i+1)+'.pkl')
	predict=multiPredict(K,Model,test_X)
	statistic(predict,test_Y)
	
	# Random Decision Forest
	K=3
	Model = []
	kf = KFold(n_splits=K)
	for train,test in kf.split(train_X):
		clf=RandomForestClassifier(n_estimators=300)
		clf.fit(train_X[train],train_Y[train])
		score = clf.score(train_X[test],train_Y[test])
		Model.append(clf)
	for i in range(K):
		joblib.dump(Model[i],'model/parameters-RF-'+str(i+1)+'.pkl')
	predict=multiPredict(K,Model,test_X)
	statistic(predict,test_Y)

	# ANN
	units=(50)
	K=3
	Model = []
	kf = KFold(n_splits=K)
	for train,test in kf.split(train_X):
		clf=MLPClassifier(hidden_layer_sizes=units,activation='logistic',solver='adam',learning_rate_init=0.0001,max_iter=2000)
		clf.fit(train_X[train],train_Y[train])
		score = clf.score(train_X[test],train_Y[test])
		Model.append(clf)
	for i in range(K):
		joblib.dump(Model[i],'model/parameters-ANN-'+str(i+1)+'.pkl')
	predict=multiPredict(K,Model,test_X)
	statistic(predict,test_Y)

	# KNN
	K=5
	Model = []
	kf = KFold(n_splits=K)
	for train,test in kf.split(train_X):
		clf = KNeighborsClassifier(algorithm='kd_tree',n_neighbors=3)
		clf.fit(train_X[train],train_Y[train])
		score = clf.score(train_X[test],train_Y[test])
		Model.append(clf)
	for i in range(K):
		joblib.dump(Model[i],'model/parameters-KNN-'+str(i+1)+'.pkl')
	predict=multiPredict(K,Model,test_X)
	statistic(predict,test_Y)

	
	Model=[]
	# SVM 0-2
	for i in range(3):
		clf=joblib.load('model/parameters-SVM-'+str(i+1)+'.pkl')
		Model.append(clf)
	# ANN 3-5
	for i in range(3):
		clf=joblib.load('model/parameters-ANN-'+str(i+1)+'.pkl')
		Model.append(clf)
	# KNN 6-10
	for i in range(5):
		clf=joblib.load('model/parameters-KNN-'+str(i+1)+'.pkl')
		Model.append(clf)
	# Random Forest 11-13
	for i in range(3):
		clf=joblib.load('model/parameters-RF-'+str(i+1)+'.pkl')
		Model.append(clf)
	
	# Logistic Regression 14-16
	for i in range(3):
		clf=joblib.load('model/parameters-LR-'+str(i+1)+'.pkl')
		Model.append(clf)
	
	K=17
	clf=RandomForestClassifier(n_estimators=500)
	
	predict=Predict(K,Model,train_X)
	clf.fit(predict,train_Y)
	print(clf.score(predict,train_Y))


	predict=Predict(K,Model,test_X)
	print(clf.score(predict,test_Y))
	clf.fit(predict,test_Y)
	print(clf.score(predict,test_Y))
	
	joblib.dump(clf,'model/parameters-final.pkl')	
