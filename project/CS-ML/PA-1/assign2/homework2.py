from numpy import *
import random
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')



#-----------------------------数据加载模块------------------------------

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
		ls=l.strip().split('	')
		for i in range(len(ls)):
			a=float(ls[i])
			_X.append(a)
		if ok==1:
			_A=[]
			for i in range(len(ls)):
				_A.append(1)
			ok=0
			X.append(_A)
		X.append(_X)
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

def Least_square_method(X,Y):
	_X=X.T*X
	_X=_X.I
	_X=(_X*(X.T))
	_X=_X*Y
	return mat(_X)


#----------------------------训练数据添加二次项-----------------------------------

def addQuadraticTerm(X):
	_X=[]
	m,n=X.shape
	for i in range(m):
		x=[]
		for j in range(1,n):
			x.append(float(X[i,j])**2)
		_X.append(x)
	#print(mat(_X))
	_X=hstack((X,mat(_X)))
	return _X


#-------------------------------------------------------------------------------

#-----------------------------------学习算法模块---------------------------------


#-----------最小二乘法求解----------------------------------
def least_squares(feature,label):
	w=(feature.T*feature).I * feature.T*label
	return w


#----------ridge regression(岭回归) 的最小二乘解法----------
def ridge_regression(feature, label, lamda):
	n=feature.shape[1]
	w=(feature.T*feature+lamda*mat(eye(n))).I * feature.T*label
	return w

#-----------------lasso regression------------------------ 
'''
	用最小二乘法求解，切除绝对值小于lamda/2的参数
	重复，直到参数个数不变
'''
def lasso_regression(feature, label, lamda):
	N=feature.shape[1]
	X=feature
	l=ones(N)
	n=N
	while True:
		w=least_squares(X,label)
		ok=0
		for i in range(n):
			if w[i,0]>=-lamda/2.0 and w[i,0]<=lamda/2.0:
				ok+=1
				w[i,0]=0
			'''
			elif w[i,0] < -lamda/2.0:
				w[i,0]+=lamda/2.0
			elif w[i,0] > lamda/2.0:
				w[i,0]-=lamda/2.0
			'''
		if ok==0 or ok==n:
			w=origin_W(l,w)
			break
		else:
			w,n,l,X=cut_matrix(w,n,l,X)
	return w
	
# 切除系数值为0的feature值
def cut_matrix(w,n,l,X):
	a=[]
	for i in range(len(l)):
		if l[i]!=0:
			a.append(i)
	_w=[]
	_l=0
	_X=[]
	m=X.shape[0]
	for i in range(n):
		if w[i,0]!=0:
			_w.append(float(w[i,0]))
			_l+=1
			x=[]
			for j in range(m):
				x.append(float(X[j,i]))
			_X.append(x)
		else:
			l[a[i]]=0
	return mat(_w).T,_l,l,mat(_X).T

# 还原原本的W
def origin_W(l,W):
	_W=[]
	a=[]
	for i in range(len(l)):
		_W.append(0)
		if l[i]!=0:
			a.append(i)
	for i in range(W.shape[0]):
		_W[a[i]]=W[i,0]
	return mat(_W).T

#----------------robust regression (稳健回归)------------------

def robust_regression(feature, label, lamda,thresh,per):
	while True:
		X,Y=dataSplit(feature,label, per)
		W=least_squares(X,Y)
		a=badRate(feature*W,label,thresh)
		if a<lamda:
			return W

def badRate(_Y,Y,thresh):
	n=Y.shape[0]
	c=0
	_Y=abs(_Y-Y)
	Y=abs(Y)
	for i in range(n):
		if _Y[i,0]>thresh:
			c+=1
	return c/n

def dataSplit(X, Y, p):
	n,m=X.shape
	np=int(n*p)
	_X=[]
	_Y=[]
	while np>0:
		np-=1
		if n==1: 
			k=0
		else:
			k=random.randint(0,n-1)
		x=[]
		for i in range(m):
			x.append(float(X[k,i]))
		_X.append(x)
		_Y.append(float(Y[k,0]))
		ok=0
		if k==0:
			A=X[1:,:]
			B=Y[1:,:]
		elif k==(n-1):
			A=X[:n-2,:]
			A=X[:n-2,:]
		else:
			A=vstack((X[:k,:],X[k+1:,:]))
			B=vstack((Y[:k,:],Y[k+1:,:]))
		X=A
		Y=B
		n-=1
	return mat(_X),mat(_Y).T

#-------------------------------------------------------------------------------

#-----------------------------------误差评估-------------------------------------

# 计算预测结果的方差
def meanSquareError(W,X,Y):
	error=0
	m=X.shape[0]
	_Y=X*W
	for i in range(m):
		error+=(_Y[i,0]-Y[i,0])**2
	return error/m

# 计算预测结果的绝对值误差
def meanAbsoluteError(W,X,Y):
	error=0
	m=X.shape[0]
	_Y=X*W
	for i in range(m):
		error+=abs(_Y[i,0]-Y[i,0])
	return error/m


#-------------------------------------------------------------------------------

#--------------------------------主函数------------------------------------------

if __name__ =="__main__":
	fx='count_data_trainx.txt'
	fy='count_data_trainy.txt'
	rawTrain_X,rawTrain_Y=load_data(fx,fy)   
	#print(train_X.shape)
	#print(train_Y.shape)

	fx='count_data_testx.txt'
	fy='count_data_testy.txt'
	test_X,test_Y=load_data(fx,fy)

#-----------------------------------task 1-------------------------------------
	
	# task1: 使用原生的9+1维训练数据

	print('')
	print('--------------------------------------------------------------')
	print('')
	print('raw data with 9 features and 1 bias:')
	print('')
	print('------------------------------------')
	print('')
	
	# least squares
	W=least_squares(rawTrain_X,rawTrain_Y)
	absoluteError=meanAbsoluteError(W,test_X,test_Y)
	squareError=meanSquareError(W,test_X,test_Y)
	print('Bayesian regression:')
	#print('coefficients W =',W.T)
	print('absolute error =',absoluteError)
	print('square error =',squareError)
	print('')
	print('------------------------------------')
	print('')
	
	# ridge regression
	W=ridge_regression(rawTrain_X,rawTrain_Y,0.5)
	absoluteError=meanAbsoluteError(W,test_X,test_Y)
	squareError=meanSquareError(W,test_X,test_Y)
	print('Bayesian regression:')
	#print('coefficients W =',W.T)
	print('absolute error =',absoluteError)
	print('square error =',squareError)
	print('')
	print('------------------------------------')
	print('')

	# lasso regression
	W=lasso_regression(rawTrain_X,rawTrain_Y,1)
	absoluteError=meanAbsoluteError(W,test_X,test_Y)
	squareError=meanSquareError(W,test_X,test_Y)
	print('Bayesian regression:')
	#print('coefficients W =',W.T)
	print('absolute error =',absoluteError)
	print('square error =',squareError)
	print('')
	print('------------------------------------')
	print('')

	# robust regression
	# 目前最佳的学习参数为 lamda=0.3 thresh=1.5 per=0.3  可自行调参
	W=robust_regression(rawTrain_X,rawTrain_Y,0.3,1.5,0.3)
	absoluteError=meanAbsoluteError(W,test_X,test_Y)
	squareError=meanSquareError(W,test_X,test_Y)
	print('Bayesian regression:')
	#print('coefficients W =',W.T)
	print('absolute error =',absoluteError)
	print('square error =',squareError)
	print('')
	print('------------------------------------')
	print('')
	
	# Bayesian regression 
	from sklearn import linear_model
	clf = linear_model.BayesianRidge()
	clf = clf.fit(rawTrain_X, rawTrain_Y)
	W= mat(clf.coef_).T
	absoluteError=meanAbsoluteError(W,test_X,test_Y)
	squareError=meanSquareError(W,test_X,test_Y)
	print('Bayesian regression:')
	#print('coefficients W =',W.T)
	print('absolute error =',absoluteError)
	print('square error =',squareError)
	print('')
	print('---------------------------------------------------------------')
	print('')
	
	

#-----------------------------------task 2-------------------------------------
	
	# task2: 使用添加了二次项的18+1维训练数据
	Train_Y=rawTrain_Y
	Train_X=addQuadraticTerm(rawTrain_X)
	test_X=addQuadraticTerm(test_X)
	#print(Train_X.shape)
	#print(Train_Y.shape)
	
	print('')
	print('--------------------------------------------------------------')
	print('')
	print('data with 18 features and 1 bias: ( Add quadratic terms )')
	print('')
	print('------------------------------------')
	print('')
	# 最小二乘法解
	W=least_squares(Train_X,Train_Y)
	absoluteError=meanAbsoluteError(W,test_X,test_Y)
	squareError=meanSquareError(W,test_X,test_Y)
	print('Bayesian regression:')
	#print('coefficients W =',W.T)
	print('absolute error =',absoluteError)
	print('square error =',squareError)
	print('')
	print('------------------------------------')
	print('')
	
	# 岭回归的最小二乘法解
	W=ridge_regression(Train_X,Train_Y,0.5)
	absoluteError=meanAbsoluteError(W,test_X,test_Y)
	squareError=meanSquareError(W,test_X,test_Y)
	print('Bayesian regression:')
	#print('coefficients W =',W.T)
	print('absolute error =',absoluteError)
	print('square error =',squareError)
	print('')
	print('------------------------------------')
	print('')

	# lasso regression
	W=lasso_regression(Train_X,Train_Y,1)
	absoluteError=meanAbsoluteError(W,test_X,test_Y)
	squareError=meanSquareError(W,test_X,test_Y)
	print('Bayesian regression:')
	#print('coefficients W =',W.T)
	print('absolute error =',absoluteError)
	print('square error =',squareError)
	print('')
	print('------------------------------------')
	print('')

	# robust regression
	# 目前最佳的学习参数为 lamda=0.3 thresh=1.5 per=0.3  可自行调参
	W=robust_regression(Train_X,Train_Y,0.3,1.5,0.3)
	absoluteError=meanAbsoluteError(W,test_X,test_Y)
	squareError=meanSquareError(W,test_X,test_Y)
	print('Bayesian regression:')
	#print('coefficients W =',W.T)
	print('absolute error =',absoluteError)
	print('square error =',squareError)
	print('')
	print('------------------------------------')
	print('')
	
	# Bayesian regression 
	from sklearn import linear_model
	clf = linear_model.BayesianRidge()
	clf = clf.fit(Train_X, Train_Y)
	W= mat(clf.coef_).T
	absoluteError=meanAbsoluteError(W,test_X,test_Y)
	squareError=meanSquareError(W,test_X,test_Y)
	print('Bayesian regression:')
	#print('coefficients W =',W.T)
	print('absolute error =',absoluteError)
	print('square error =',squareError)
	print('')
	print('------------------------------------')
	print('')
	