from numpy import *
import random
import matplotlib.pyplot as plt

def load_data(file_name):  
	#load training data from file and generate feature matrix X and feature matrix Y
	f = open(file_name,'r')
	X=[]
	for l in f.readlines():
		_X=[]
		ls=l.strip().split('\t')
		for i in range(len(ls)):
			_X.append(float(ls[i]))
		X.append(_X)
	f.close()
	return mat(X).T


def show_plot(C,X,Y):
	#show the hyperplane plot and distribution of all points
	data=array(X)
	c1_X1=[]
	c1_X2=[]
	c2_X1=[]
	c2_X2=[]
	c3_X1=[]
	c3_X2=[]
	c4_X1=[]
	c4_X2=[]
	n = shape(data)[0]
	for i in range(n):
		if(int(Y[i,0])==1):
			c1_X1.append(X[i,0])
			c1_X2.append(X[i,1])
		elif(int(Y[i,0])==2):
			c2_X1.append(X[i,0])
			c2_X2.append(X[i,1])
		elif(int(Y[i,0])==3):
			c3_X1.append(X[i,0])
			c3_X2.append(X[i,1])
		elif(int(Y[i,0])==0 or int(Y[i,0])==4):
			c4_X1.append(X[i,0])
			c4_X2.append(X[i,1])

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(c1_X1, c1_X2, s=10, c='BLUE')
	ax.scatter(c2_X1, c2_X2, s=10, c='RED')
	ax.scatter(c3_X1, c3_X2, s=10, c='YELLOW')
	ax.scatter(c4_X1, c4_X2, s=10, c='GREEN')
	
	ax.scatter(C[1,0],C[1,1], marker='x',s=30,c='BLUE')
	ax.scatter(C[2,0],C[2,1], marker='x',s=30,c='RED')
	ax.scatter(C[3,0],C[3,1], marker='x',s=30,c='YELLOW')
	ax.scatter(C[0,0],C[0,1], marker='x',s=30,c='GREEN')
	'''
	ax.scatter(C[1,0],C[1,1], marker='x',s=30)
	ax.scatter(C[2,0],C[2,1], marker='x',s=30)
	ax.scatter(C[3,0],C[3,1], marker='x',s=30)
	ax.scatter(C[0,0],C[0,1], marker='x',s=30)
	'''
	plt.xlabel('X1');
	plt.ylabel('X2')
	plt.show()

#-----------K-means clustring algorithom--------------------
def distance(vecA, vecB):
	'''欧氏距离
	input: vecA(mat), vecB(mat): two vector
	output: dist[0,0](float)
	'''
	dist = (vecA-vecB) * (vecA-vecB).T
	return float(dist[0,0])

def randCent(data, k):
	'''初始化k个聚类中心
	input: data(mat), k(int)
	output: centroids(mat): k个聚类中心的坐标向量
	'''
	n=data.shape[1] #feature的维度
	centroids=mat(zeros((k,n))) #初始化k个聚类中心，皆为(0,0,...,0)
	for i in range(n):
		Min=min(data[:,i])
		Range=max(data[:,i])-Min
		for j in range(k):
			centroids[j,i]=float(Min)+random.uniform(0,1)*Range
	return centroids

def k_means(data,k,centroids):
	'''K-means 聚类算法求解k个聚类中心
	input: 	data(mat): feature
			k(int): 设定划分成k个聚类
			centroids(mat): 初始化的k个聚类中心
	output:	centroids(mat): 训练完成的k个聚类中心
			subCenter(mat): 每个样本所处的类别	
	'''
	m=data.shape[0]
	n=data.shape[1]
	subCenter=mat(zeros((m,2)))
	change=True
	while change==True:
		change=False
		for i in range(m):
			minDist=inf
			minIndex=0
			for j in range(k):
				dist=distance(data[i,],centroids[j,])
				if dist<minDist:
					minDist=dist
					minIndex=j
			if subCenter[i,0] != minIndex:
				change=True
				subCenter[i,]=mat([minIndex,minDist])
		#重新计算聚类中心
		for j in range(k):
			sum_all=mat(zeros((1,n)))
			r=0
			for i in range(m):
				if subCenter[i,0]==j:
					sum_all+=data[i,]
				r+=1
			for i in range(n):
				try:
					centroids[j,i]=sum_all[0,i]/r
				except:
					print("r is zero")
	return centroids,subCenter


#-----------------------------------------------------------

def err_rate(_Y,Y):
	m = Y.shape[0]
	err=0
	for i in range(m):
		if int(_Y[i,0])!=int(Y[i,0]):
			err+=1
	return err/m


#-----------------------------------------------------------

if __name__ =="__main__":
	fx='cluster_data_dataA_X.txt'  
	fy='cluster_data_dataA_Y.txt'
	X=load_data(fx) 
	label=load_data(fy)
	cents=randCent(X,4)
	#print(cents)
	#show_plot(cents,X,label)
	cents,train_Y=k_means(X,4,cents)
	#print(cents)
	#print(train_Y)
	err=err_rate(train_Y,label)
	print(err)
	show_plot(cents,X,train_Y)
	
	
	#print(cents)
	#print(X) 
	#print(label) 
	














