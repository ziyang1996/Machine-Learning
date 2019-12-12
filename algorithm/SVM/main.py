from numpy import *
import SVM
import matplotlib.pyplot as plt

#---------------------------------------------加载处理训练样本----------------------------------------------------

def load_data(file_name):  
	#load training data from file and generate feature matrix X and feature matrix Y
	f = open(file_name,'r')
	X=[]
	Y=[]
	for l in f.readlines():
		_X=[]
		_Y=[]
		ls=l.strip().split('\t')
		for i in range(len(ls)-1):
			_X.append(float(ls[i]))
		_Y.append(int(ls[-1]))
		X.append(_X)
		Y.append(_Y)
	f.close()
	return mat(X), mat(Y)

def transformLabel(Y):
	m = Y.shape[0]
	_Y = []
	for i in range(m):
		if int(Y[i,0])>0:
			_Y.append(1)
		else:
			_Y.append(-1)
	return mat(_Y).T



#-------------------------------------------------得到预测结果----------------------------------------------------

def get_predict(svm, X):
	m = X.shape[0]
	predict = []
	for i in range(m):
		pre_y = SVM.svm_predict(svm, X[i,:])
		predict.append(sign(pre_y[0,0]))
	return mat(predict).T


#-------------------------------------------------生成预测图像----------------------------------------------------

def show_plot(X,Y):
	#show the hyperplane plot and distribution of all points
	c1_X1=[]
	c1_X2=[]
	c2_X1=[]
	c2_X2=[]
	n = X.shape[0]
	for i in range(n):
		if int(Y[i,0])>0:
			c1_X1.append(X[i,0])
			c1_X2.append(X[i,1])
		else:
			c2_X1.append(X[i,0])
			c2_X2.append(X[i,1])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(c1_X1, c1_X2, marker="o", s=10, c='BLUE')
	ax.scatter(c2_X1, c2_X2, marker="x",s=10, c='RED')
	plt.xlabel('X1');
	plt.ylabel('X2')
	plt.show()



#-------------------------------------------------归一化特征值(normalization)---------------------------------------

def Normalization(X):
	_X=[]
	m,n=X.shape
	Min=[]
	Max=[]

	for i in range(n):
		Min.append(float(min(X[:,i])))
		Max.append(float(max(X[:,i])))
	
	for i in range(m):
		_x=[]
		for j in range(n):
			_x.append( ((X[i,j] - Min[j]) / (Max[j]-Min[j]) - 0.5)*2 ) 
			# 此处将feature值归一化压缩到 [-1,1] 区间内
		_X.append(_x)
	return mat(_X)


#--------------------------------------------------主函数---------------------------------------------------------


if __name__ == "__main__":
	# 1. 导入训练数据
	f='train.txt'
	raw_X,raw_Y = load_data(f)
	#print(raw_X.shape,raw_Y.shape)
	dataSet = Normalization(raw_X)
	labels = transformLabel(raw_Y)
	
	
	# 2. 训练SVM模型
	C = 1.2
	toler = 0.001
	svm_model = SVM.SVM_training(dataSet, labels, C, toler)

	# 3. 计算训练的准确性
	accuracy = SVM.cal_accuracy(svm_model, dataSet, labels)
	predict = get_predict(svm_model, dataSet)
	
	print("accuracy =", accuracy)
	print(predict.shape)
	show_plot(raw_X,predict)
	

