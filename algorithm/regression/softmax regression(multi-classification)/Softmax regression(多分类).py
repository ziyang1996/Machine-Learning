from numpy import *
import random
import math


def load_data(fileX,fileY):  
	#load training data from file and generate feature matrix X and feature matrix Y
	fx = open(fileX,'r')
	fy = open(fileY,'r')
	X=[]
	Y=[]
	n=0
	ok=1
	for l in fx.readlines():
		_X=[]
		_x=[]
		ls=l.strip().split('	')
		for i in range(len(ls)):
			a=float(ls[i])
			_X.append(a)
			_x.append(a*a)  #添加二次项
		if ok==1:
			_A=[]
			for i in range(len(ls)):
				_A.append(1)
			ok=0
			X.append(_A)
		X.append(_X)
		X.append(_x)
	fx.close()

	for l in fy.readlines():
		Y.append(float(l))
	fy.close()
	return mat(X).T,mat(Y).T

def standard_variance(_Y,Y):
	n=Y.shape[0]
	e=(Y-_Y)
	E=math.sqrt(e.T*e/n)
	return E

def gradientAscent(X,Y,k,maxCycle,alpha):
	m,n=X.shape
	W=mat(ones((n,k)))
	for i in range(axCycle):
		err = exp(X*W)
		rowsum = -err.sum(axis=1)
		rowsum = rowsum.repeat(k,axis=1)
		err = err / rowsum
		for x in range(m):
			err[x,Y[x,0]]+=1
		W=W+(alpha/m)*X.T*err
	return W




if __name__ =="__main__":
	'''
	fx='count_data_trainx.txt'
	fy='count_data_trainy.txt'
	train_X,train_Y=load_data(fx,fy)   

	fx='count_data_testx.txt'
	fy='count_data_testy.txt'
	test_X,test_Y=load_data(fx,fy)
	#print(train_X)
	#print(train_X.shape[0],train_X.shape[1])
	'''
