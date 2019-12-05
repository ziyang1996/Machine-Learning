
from numpy import *
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
import os
import warnings
import copy
warnings.filterwarnings('ignore')

#--------------------------------load data---------------------------------

def loadDataLabel(file):
	f=open(file,"r")
	X=[]
	for l in f.readlines():
		_X=[]
		for i in range(len(l)):
			if l[i].isdigit():
				_X.append(int(l[i]))
		X.append(_X)
	return mat(X)

def loadDataFeature(file):
	f=open(file,"r")
	X=[]
	for l in f.readlines():
		l=l.split('\t')
		#print(l)
		_X=[]
		for i in range(len(l)):
			_X.append(int(l[i]))
		X.append(_X)
	return mat(X)


#--------------------------------save data---------------------------------

def saveData(X,file):
	m,n=X.shape
	f=open(file,'w+')
	for i in range(m):
		for j in range(n):
			f.write(str(X[i,j]))
			f.write(' ')
		f.write('\n')
	f.close()

def graphSharpen(X):
	m,n=X.shape
	_X=copy.copy(X)
	for i in range(m):
		for j in range(n):
			if X[i,j]>100:
				_X[i,j]=1
			else:
				_X[i,j]=0
	return _X


def pointCompression(X):
	m,n=X.shape
	a=[0]*784
	for i in range(m):
		for j in range(784):
			if X[i,j]!=0:
				a[j]+=1
	blank=[]
	for i in range(784):
		if a[i]<5:
			blank.append(i)
	return blank
	

def imageCompression(X,blank):
	_X=[]
	m,n=X.shape
	for i in range(m):
		_x=[]
		for j in range(n):
			if j not in blank:
				_x.append(X[i,j])
		_X.append(_x)
	return mat(_X)


#--------------------------------main function-----------------------------
if __name__ =="__main__":
	fx='data/digits4000_digits_vec.txt'
	fy='data/digits4000_digits_labels.txt'
	feature=loadDataFeature(fx)
	label=loadDataLabel(fy)
	blank=[]
	images=graphSharpen(feature)
	blank=pointCompression(images)
	print(len(blank))
	train_X=imageCompression(images,blank)
	print(train_X.shape)
	
	X_train, X_test, y_train, y_test = train_test_split(train_X,label,test_size=0.1, random_state=9)
	f='data/train_feature.txt'
	saveData(X_train,f)
	f='data/train_label.txt'
	saveData(y_train,f)
	f='data/test_feature.txt'
	saveData(X_test,f)
	f='data/test_label.txt'
	saveData(y_test,f)
	
	
	fx='data/cdigits_digits_vec.txt'
	fy='data/cdigits_digits_labels.txt'
	feature_c=loadDataFeature(fx)
	label_c=loadDataLabel(fy)
	images_c=graphSharpen(feature_c)
	test_c=imageCompression(images_c,blank)
	print(test_c.shape)
	f='data/test_feature-c.txt'
	saveData(test_c,f)
	f='data/test_label-c.txt'
	saveData(label_c,f)
	