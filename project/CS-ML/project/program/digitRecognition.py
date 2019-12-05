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
	errNum1=[0]*10
	errNum2=[0]*10
	sumNum1=[0]*10
	sumNum2=[0]*10
	Num=[[0]*10]*10
	Num=mat(Num)
	m=Y.shape[0]
	for i in range(m):
		if int(_Y[i,0])!=int(Y[i,0]):
			errNum1[int(Y[i,0])]+=1
			errNum2[int(_Y[i,0])]+=1
			Error+=1
			Num[int(Y[i,0]),int(_Y[i,0])]+=1
		sumNum1[int(Y[i,0])]+=1
		sumNum2[int(_Y[i,0])]+=1
		Sum+=1
	print('------------------------------------')
	print('error statistic: ')
	print('sum digits: ',Sum)
	print('accuracy: ',round((1.0-Error/Sum)*100.0,3),'%')
	for i in range(10):
		print(i,': ',end='\t')
		print(round(errNum1[i]/sumNum1[i]*100.0,2),'%',end='\t\t')
		print(round(errNum2[i]/sumNum2[i]*100.0,2),'%')
	print('------------------------------------')
	print('\t',end='')
	for i in range(10):
		print(i,end='\t')
	print('')
	for i in range(10):
		print(i,end='\t')
		for j in range(10):
			print(Num[i,j],end='\t')
		print('')

#---------------------------multi model prediction--------------------------
def multiPredict(K,Model,X):
	predict=[]
	for i in range(K):
		pre=Model[i].predict(X)
		predict.append(pre)
	predict=mat(predict).T
	return predict
	



#--------------------------------main function-----------------------------
if __name__ =="__main__":
	fx='data/test_feature-c.txt'
	fy='data/test_label-c.txt'
	test_Xc=loadData(fx)
	test_Yc=loadData(fy)

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
	

	clf=joblib.load('model/parameters-final.pkl')

	preXC=multiPredict(17,Model,test_Xc)
	predict=mat(clf.predict(preXC)).T
	statistic(predict,test_Yc)
