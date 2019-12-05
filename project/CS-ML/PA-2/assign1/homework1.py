from numpy import *
import random
import matplotlib.pyplot as plt

#-----------------------------数据加载和图像显示模块------------------------------

def load_data(file_name):  
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
	m = X.shape[0]
	M = {}
	k = 0
	for i in range(m):
		if int(Y[i,0]) not in M:
			M[int(Y[i,0])]=k	
			k+=1
	n = len(M)
	
	X1 = [[] for i in range(k)]
	X2 = [[] for i in range(k)]
	
	for i in range(m):
		X1[M[int(Y[i,0])]].append(float(X[i,0]))
		X2[M[int(Y[i,0])]].append(float(X[i,1]))
	
	plt.figure()
	for i in range(k):
		plt.scatter(X1[i],X2[i],s=5)
	plt.xlabel('X1');
	plt.ylabel('X2')
	plt.show()
	

#-----------------------------------------------------------------------------------

#-----------------------------------学习算法模块-------------------------------------


#-----------K-means clustring algorithom-----------------------
def distance(vecA, vecB):
	'''Euclidean Distance
	input: vecA(mat), vecB(mat): two vector
	output: dist[0,0](float)
	'''
	dist = (vecA-vecB) * (vecA-vecB).T
	return float(dist[0,0])

def randCent(data, k):
	''' initial k cluster centers
	input: data(mat), k(int)
	output: centroids(mat)
	'''
	n=data.shape[1] 
	centroids=mat(zeros((k,n))) 
	for i in range(n):
		Min=min(data[:,i])
		Range=max(data[:,i])-Min
		for j in range(k):
			centroids[j,i]=float(Min)+random.uniform(0,1)*Range
	return centroids

def k_means(data, k):
	'''K-means 
	input: 	data(mat): feature
			k(int): target number of clusters
			centroids(mat): initial k cluster centers
	output:	centroids(mat): learned k cluster centers
			subCenter(mat): learned label of samples 	
	'''
	centroids=randCent(data,k)

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
		# recalculate centroids
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
	return subCenter



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

def meanShift(points, kernel_bandwidth=2, alpha=0.001):
	''' train Mean Shift model
	input:	points(array): features data
			kenel_bandwidth(int)
			MIN_DISTANCE()
	output:	points(mat): centroids
			mean_shift_points(mat)
			group(array): clusters
	'''
	MIN_DISTANCE = alpha
	mean_shift_points = copy(points)
	max_min_dist = 1
	iteration = 0 
	m = mean_shift_points.shape[0]
	need_shift = [True] * m

	while max_min_dist > MIN_DISTANCE:
		#print("iteration : ",str(iteration)," dist = ",max_min_dist)
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

	return mat(group).T


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


#-------------------------------------------------------------------------------

#--------------------------------主函数------------------------------------------

if __name__ =="__main__":
	
	fx='cluster_data_dataC_X.txt'  
	fy='cluster_data_dataC_Y.txt'
	feature=load_data(fx) 
	label=load_data(fy)

	# K-means algorithm
	cluster = k_means(feature,4)
	show_plot(feature,cluster)

	# EM-GMM
	from sklearn.mixture import GaussianMixture
	clf = GaussianMixture(n_components=4,max_iter=100,random_state=1)
	clf.fit(feature)
	cluster=mat(clf.predict(feature)).T
	show_plot(feature,cluster)

	# mean-shift algorithm
	cluster = meanShift(feature,3,0.001)
	show_plot(feature,cluster)