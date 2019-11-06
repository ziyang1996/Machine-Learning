from numpy import *
import random
import math
from copy import *
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


def show_plot(X,Y):
	#show the hyperplane plot and distribution of all points
	c1_X1=[]
	c1_X2=[]
	c2_X1=[]
	c2_X2=[]
	c3_X1=[]
	c3_X2=[]
	c4_X1=[]
	c4_X2=[]
	n = shape(X)[0]
	for i in range(n):
		if(int(Y[i,0])==1):
			c1_X1.append(float(X[i,0]))
			c1_X2.append(float(X[i,1]))
		elif(int(Y[i,0])==2):
			c2_X1.append(float(X[i,0]))
			c2_X2.append(float(X[i,1]))
		elif(int(Y[i,0])==3):
			c3_X1.append(float(X[i,0]))
			c3_X2.append(X[i,1])
		elif(int(Y[i,0])==0 or int(Y[i,0])==4):
			c4_X1.append(float(X[i,0]))
			c4_X2.append(float(X[i,1]))

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(c1_X1, c1_X2, s=10, c='BLUE')
	ax.scatter(c2_X1, c2_X2, s=10, c='RED')
	ax.scatter(c3_X1, c3_X2, s=10, c='YELLOW')
	ax.scatter(c4_X1, c4_X2, s=10, c='GREEN')
	'''
	ax.scatter(C[1,0],C[1,1], marker='x',s=30,c='BLUE')
	ax.scatter(C[2,0],C[2,1], marker='x',s=30,c='RED')
	ax.scatter(C[3,0],C[3,1], marker='x',s=30,c='YELLOW')
	ax.scatter(C[0,0],C[0,1], marker='x',s=30,c='GREEN')
	
	ax.scatter(C[1,0],C[1,1], marker='x',s=30)
	ax.scatter(C[2,0],C[2,1], marker='x',s=30)
	ax.scatter(C[3,0],C[3,1], marker='x',s=30)
	ax.scatter(C[0,0],C[0,1], marker='x',s=30)
	'''
	plt.xlabel('X1');
	plt.ylabel('X2')
	plt.show()


#-----------------------------------------------------------

def err_rate(_Y,Y):
	m = Y.shape[0]
	err=0
	for i in range(m):
		if int(_Y[i,0])!=int(Y[i,0]):
			err+=1
	return err/m



#-----------Mean Shift clustring algorithom--------------------

def DBSCAN(data, eps, MinPts):
	''' DBSCAN 算法
	input:	data(mat)
			eps(float): radius
			MinPts(int): least number of points in radius
	output: types(mat): point types(core, boundary or noise)
			sub_class(mat): classification results
	'''
	m = data.shape[0]
	types = mat(zeros((1,m)))
	sub_class = mat(zeros((1,m)))
	dealed = mat(zeros((m,1))) # visit flag

	dis = distance(data)

	number = 1
	for i in range(m):
		if dealed[i,0] == 0:
			D = dis[i, ]
			ind = find_eps(D, eps)
			# 边界点
			if len(ind) > 1 and len(ind) < MinPts+1:
				types[0,i] = 0
				sub_class[0,i] = 0 
			# 噪音点
			if len(ind) == 1:
				types[0,i] = -1
				sub_class[0,i] = -1
				dealed[i,0] = 1 
			# 核心点
			if len(ind) >= MinPts+1:
				types[0,i] = 1 
				for x in ind:
					sub_class[0,x] = number
				# 判断核心点是否密度可达
				while len(ind) > 0:
					dealed[ind[0],0] = 1
					D = dis[ind[0], ]
					tmp = ind[0]
					del ind[0]
					ind_1 = find_eps(D,eps)
					# 处理非噪音点
					if len(ind_1) > 1: 
						for x1 in ind_1:
							sub_class[0,x1] = number
						if len(ind_1) >= MinPts + 1:
							types[0,tmp] = 1
						else:
							types[0,tmp] = 0 

						for j in range(len(ind_1)):
							if dealed[ind_1[j], 0] == 0:
								dealed[ind_1[j], 0] = 1
								ind.append(ind_1[j])
								sub_class[0, ind_1[j]] = number
				number += 1 
	# 最后将所有未分类的点设置为噪声点
	ind_2 = ((sub_class == 0).nonzero())[1]
	for x in ind_2:
		sub_class[0,x] = -1 
		types[0, x] = -1 

	return types.T, sub_class.T


def find_eps(distance_D, eps):
	''' 搜集所有距离小于等于eps的样本
	input:	distance_D(mat)： 样本i和其他样本之间的距离
			eps(float): 半径大小
	output:	ind(list): 距离小于等于eps的样本下标
	'''
	ind = []
	n = distance_D.shape[1]
	for j in range(n):
		if distance_D[0,j] <= eps:
			ind.append(j)
	return  ind


def euclidean_distance(pointA, pointB):
	'''欧氏距离
	input: pointA(mat), pointB(mat): two vector
	output: dist[0,0](float)
	'''
	dist = (pointA-pointB) * (pointA-pointB).T
	return math.sqrt(dist)


def distance(data):
	dis=[]
	m, n = data.shape
	for i in range(m):
		_dis=[]
		for j in range(m):
			d=euclidean_distance(data[i,],data[j,])
			_dis.append(d)
		dis.append(_dis)
	return mat(dis)

def epsilon(data, MinPts):
	''' 计算半径
	input: 	data(mat): 训练数据
			MinPts(int): 半径内的数据点个数
	output: eps(float): 半径
	'''
	m, n = data.shape
	xMax = data.max(0)
	xMin = data.min(0)
	eps = ((prod(xMax - xMin) * MinPts * math.gamma(0.5 * n + 1)) / (m * math.sqrt(math.pi ** n))) ** (1.0 / n)
	return eps

#-----------------------------------------------------------

if __name__ =="__main__":
	fx='cluster_data_dataC_X.txt'  
	fy='cluster_data_dataC_Y.txt'
	data=load_data(fx) 
	label=load_data(fy)
	show_plot(data,label)
	MinPts = 3
	eps = epsilon(data, MinPts)
	print(eps)

	types, sub_class = DBSCAN(data, eps, MinPts)
	print(types)
	print(sub_class)

	show_plot(data,sub_class)
	












