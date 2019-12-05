'''
Task 1: plot a scatter plot of two features for 10 labels(0-9)
'''

from numpy import *
import matplotlib.pyplot as plt 
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
	ax.set_ylabel('Symmetry')
	ax.set_title('number:'+str(A))
	plt.show()

def plot_each(X,Y):
	m,n = X.shape
	x1 = [[] for i in range(10)]
	x2 = [[] for i in range(10)]
	for i in range(m):
		a = int(Y[i,0])
		x1[a].append(float(X[i,0]))
		x2[a].append(float(X[i,1]))
	
	plt.figure()
	
	for i in range(10):
		plt.subplot(2,5,i+1)
		plt.scatter(x1[i],x2[i],s=3)
		plt.xlabel('Intensity')
		plt.ylabel('Symmetry')
		plt.title('number:'+str(i))
		plt.xlim([0.0,0.8])
		plt.ylim([-7.0,0.0])
	plt.show()


#--------------------------------main function--------------------------
if __name__ =="__main__":
	f='data/train_features.txt'
	train_X,train_Y=load_data(f)   
	#print(train_X)
	#print(train_Y)
	# show all digit data points in one plot
	num=arange(0,10,1)
	plot(train_X,train_Y,num)

	# show only digit 1 and 5 data points in one plot
	plot_each(train_X,train_Y)
	











