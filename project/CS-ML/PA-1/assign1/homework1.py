from numpy import *
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')



#-----------------------------数据加载和图像显示模块------------------------------

def load(fileZ):
	z=open(fileZ,'r')
	Z=[]
	for l in z.readlines():
		Z.append(float(l))
	return mat(Z).T

def load_data(file_X,file_Y):
	fx=open(file_X,"r")
	fy=open(file_Y,"r")
	X=[]
	Y=[]
	for l in fx.readlines():
		l=l.strip().split('	')
		for i in range(len(l)):
			X.append(float(l[i]))

	for l in fy.readlines():
		Y.append(float(l))

	return mat(X).T,mat(Y).T


# 添加多项式的2~k次项
def poly(X,k):
	_X=[]
	for i in range(X.shape[0]):
		s=[]
		for j in range(k+1):
			s.append(math.pow(X[i],j))
		_X.append(s)
	return mat(_X)

# 显示图像
def show_plot(X,Y,W,Z):
	'''
	input: 	X(mat): feature值（转化为多项式后的）
			Y(mat): 学习得出的label值
			W(mat): 学习出的W参数向量
			Z(mat): 真正的参数向量
	'''
	X=array(X)
	Y=array(Y)
	_X=arange(-2,2,0.05)
	_Y=[]
	for i in range(len(_X)):
		a=_X[i]
		b=float(W[0])+a*float(W[1])+a*a*float(W[2])+a*a*a*float(W[3])+math.pow(a,4)*float(W[4])+math.pow(a,5)*float(W[5])
		_Y.append(b)

	_zx=arange(-2,2,0.05)
	_zy=[]
	for i in range(len(_zx)):
		a=_zx[i]
		b=float(Z[0])+a*float(Z[1])+a*a*float(Z[2])+a*a*a*float(Z[3])+math.pow(a,4)*float(Z[4])+math.pow(a,5)*float(Z[5])
		_zy.append(b)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(X,Y,color='r')
	ax.plot(_X,_Y,'b')
	ax.plot(_zx,_zy,'#ff00ff')
	plt.show()


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

# 计算预测结果的方差 （若想改成均方差，返回值开根即可）
def getError(W,X,Y):
	error=0
	m=X.shape[0]
	_Y=X*W
	for i in range(m):
		error+=(_Y[i,0]-Y[i,0])**2
	return error/m



#-------------------------------------------------------------------------------

#--------------------------------主函数------------------------------------------
if __name__ =="__main__":
	# 读取训练数据
	fx='polydata_data_sampx.txt'
	fy='polydata_data_sampy.txt'
	raw_X,Y_train=load_data(fx,fy)   
	X_train=poly(raw_X,5)  # 一维特征值转化为k次的多项式
	#print(X_train.shape)
	#print(Y_train.shape)
	
	# 读取真正的参数
	fz='polydata_data_thtrue.txt'
	Z=load(fz)
	print('ture coefficients =',Z)
	# 读取测试数据
	fx='polydata_data_polyx.txt'
	fy='polydata_data_polyy.txt'
	rawTest_X,test_Y=load_data(fx,fy)
	test_X=poly(rawTest_X,5)
	#print(test_X.shape)
	#print(test_Y.shape)

	
	# 划分训练数据的比例(题目要求改变比例观察结果)
	train_X, A, train_Y, B = train_test_split(X_train,Y_train, test_size=0.1, random_state=0)


	# least square
	W=least_squares(train_X,train_Y)
	error=getError(W,test_X,test_Y)
	print('Least Squares:')
	print('coefficients W =',W.T)
	print('error =',error)
	print('')
	print('---------------------------------------------------------------')
	print('')
	
	# ridge regression
	W=ridge_regression(train_X,train_Y,0.5)
	error=getError(W,test_X,test_Y)
	print('Ridge Regression:')
	print('coefficients W =',W.T)
	print('error =',error)
	print('')
	print('---------------------------------------------------------------')
	print('')

	# lasso regression
	W=lasso_regression(train_X,train_Y,1)
	error=getError(W,test_X,test_Y)
	print('Lasso Regression:')
	print('coefficients W =',W.T)
	print('error =',error)
	print('')
	print('---------------------------------------------------------------')
	print('')

	# robust regression
	# 目前最佳的学习参数为 lamda=0.3 thresh=1.5 per=0.3  可自行调参
	W=robust_regression(train_X,train_Y,0.3,1.5,0.3)
	error=getError(W,test_X,test_Y)
	print('Robust Regression:')
	print('coefficients W =',W.T)
	print('error =',error)
	print('')
	print('---------------------------------------------------------------')
	print('')
	
	# Bayesian regression 
	from sklearn import linear_model
	clf = linear_model.BayesianRidge()
	clf = clf.fit(train_X, train_Y)
	W= mat(clf.coef_).T
	error=getError(W,test_X,test_Y)
	print('Bayesian regression:')
	print('coefficients W =',W.T)
	print('error =',error)
	print('')
	print('---------------------------------------------------------------')
	print('')

	show_plot(rawTest_X,test_Y,W,Z)
	