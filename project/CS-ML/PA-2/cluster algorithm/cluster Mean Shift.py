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

#-----------Mean Shift clustring algorithom--------------------
MIN_DISTANCE = 0.001 # 最小误差

def euclidean_distance(pointA, pointB):
	'''欧氏距离
	input: pointA(mat), pointB(mat): two vector
	output: dist[0,0](float)
	'''
	dist = (pointA-pointB) * (pointA-pointB).T
	return math.sqrt(dist)

def gaussian_kernel(distance, bandwidth):
	''' 高斯核函数
	input: 	distance(mat): 欧氏距离
			bandwidth(int): 核函数的带宽
	output:	gaussian_val(mat): 高斯函数值
	'''
	m = distance.shape[0]
	right = mat(zeros((m, 1)))
	for i in range(m):
		right[i,0] = (-0.5 * distance[i] * distance[i].T ) / (bandwidth * bandwidth)
		right[i,0] = exp(right[i,0])
	left = 1 / (bandwidth * math.sqrt(2 * math.pi))
	gaussian_val = left * right
	return gaussian_val

def train_mean_shift(points, kernel_bandwidth=2):
	''' train Mean Shift model
	input:	points(array): features data
			kenel_bandwidth(int)
			MIN_DISTANCE()
	output:	points(mat): centroids
			mean_shift_points(mat)
			group(array): clusters
	'''
	mean_shift_points = copy(points)
	max_min_dist = 1
	iteration = 0 
	m = mean_shift_points.shape[0]
	need_shift = [True] * m

	while max_min_dist > MIN_DISTANCE:
		print("iteration : ",str(iteration)," dist = ",max_min_dist)
		max_min_dist = 0
		iteration += 1
		for i in range(m):
			if not need_shift[i]:
				continue
			p_new = mean_shift_points[i]
			p_new_start = p_new 
			p_new = shift_point(p_new, points, kernel_bandwidth)
			dist = euclidean_distance(p_new, p_new_start)

			if dist > max_min_dist:
				max_min_dist = dist
			if dist < MIN_DISTANCE:
				need_shift[i] = False
			mean_shift_points[i] = p_new
	group = group_points(mean_shift_points)

	return mat(points), mean_shift_points, mat(group).T


def shift_point(point, points, kernel_bandwidth):
	''' 计算均值漂移点
	input:	point(mat): 需要计算的点
			points(array): 所有样本点
			kernel_bandwidth(int): 核函数的带宽
	output: point_shifted(mat): 漂移后的点
	'''
	points = mat(points)
	m = points.shape[0]
	# distance
	point_distances = mat(zeros((m,1)))
	for i in range(m):
		point_distances[i,0] = euclidean_distance(point, points[i])
	# Gaussian kernel
	point_weights = gaussian_kernel(point_distances, kernel_bandwidth)
	# denominator
	all_sum = 0.0 
	for i in range(m):
		all_sum += point_weights[i,0]
	# mean shift
	point_shifted = point_weights.T * points / all_sum
	return point_shifted


def group_points(mean_shift_points):
	''' 计算所属的类别
	input: 	mean_shift_points(mat)
	output: group_assignment(array)
	'''
	group_assignment = []
	m, n = mean_shift_points.shape
	index = 0 
	index_dict = {}
	for i in range(m):
		item = []
		for j in range(n):
			item.append(str(("%5.2f" % mean_shift_points[i,j])))
		item_1 = " ".join(item)
		if item_1 not in index_dict:
			index_dict[item_1] = index 
			index+=1 

	for i in range(m):
		item = []
		for j in range(n):
			item.append(str(("%5.2f" % mean_shift_points[i,j])))
		item_1 = " ".join(item)
		group_assignment.append(index_dict[item_1])

	return group_assignment

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
	fx='cluster_data_dataC_X.txt'  
	fy='cluster_data_dataC_Y.txt'
	data=load_data(fx) 
	label=load_data(fy)
	#show_plot(data,label)
	X=data

	MIN_DISTANCE = 0.0001
	points, shift_points, cluster = train_mean_shift(X,2)
	
	show_plot(data,cluster)
	












