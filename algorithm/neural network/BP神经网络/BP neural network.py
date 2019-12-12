from numpy import *
import matplotlib.pyplot as plt

'''-------------------------------------------BP神经网络--------------------------------------------------
训练BP神经网络模型（back propaganda neural network）

分类问题的标签值： 输出层节点数 = 分类标签种类数
					exp: 第0类： [1, 0 , ...]
单一隐含层
训练函数： bp_train(feature,label,n_hidden,n_output,maxCycle,alpha)
				return w0,b0,w1,b1
预测函数： get_predict(feature, w0, w1, b0, b1)
				return _Y(mat) 预测值矩阵

'''
#-----------------------------------------训练部分-------------------------------------------------------

def bp_train(feature,label,n_hidden,n_output,maxCycle=1000,delta=0.01,learnRate=0.1):
	''' 训练模型
		input: 	feature(mat): 特征值 
				label(mat): 标签值
				n_hidden(int): 隐含层的神经元个数
				n_output(int): 输出层的神经元个数
				maxCycle(int): 最大迭代次数
				alpha(float): 学习率
		output: w0(mat): 输入层到隐含层之间的权重
				b0(mat): 输入层到隐含层之间的偏置
				w1(mat): 隐含层到输出层之间的权重
				b1(mat): 隐含层到输出层之间的偏置
	'''
	# 1 初始化神经元参数
	m,n = feature.shape
	w0 = mat(random.rand(n,n_hidden))
	w0 = w0 * (8.0*sqrt(6)/sqrt(n+n_hidden)) - mat(ones((n,n_hidden))) * (4.0*sqrt(6)/sqrt(n+n_hidden))
	b0 = mat(random.rand(1,n_hidden))
	b0 = b0 * (8.0*sqrt(6)/sqrt(n+n_hidden)) - mat(ones((1,n_hidden))) * (4.0*sqrt(6)/sqrt(n+n_hidden))
	w1 = mat(random.rand(n_hidden,n_output))
	w1 = w1 * (8.0*sqrt(6)/sqrt(n_hidden+n_output)) - mat(ones((n_hidden,n_output))) * (4.0*sqrt(6)/sqrt(n_hidden+n_output))
	b1 = mat(random.rand(1,n_output))
	b1 = b1 * (8.0*sqrt(6)/sqrt(n_hidden+n_output)) - mat(ones((1,n_output))) * (4.0*sqrt(6)/sqrt(n_hidden+n_output))
	
	# 设置递减的学习率
	#alpha=1.0/learnRate

	# 2 训练
	for i in range(0,maxCycle+1):
		# 2.1 正向传播
		# 2.1.1 计算隐含层的输入
		hidden_input = hidden_in(feature, w0, b0)
		# 2.1.2 计算隐含层的输出
		hidden_output = hidden_out(hidden_input)
		# 2.1.3 计算输出层的输入
		output_in = predict_in(hidden_output,w1,b1)
		# 2.1.4 计算输出层的输出
		output_out = predict_out(output_in)

		# 2.2 误差的反向传播
		# 2.2.1 隐含层到输出层之间的残差
		delta_output = - multiply( (label - output_out), partial_sig(output_in))
		# 2.2.2 输入层到隐含层之间的残差
		delta_hidden = multiply( (delta_output * w1.T), partial_sig(hidden_input))

		# 2.3 修正权重和偏置
		w1 = w1 - learnRate * (hidden_output.T * delta_output)
		b1 = b1 - learnRate * sum(delta_output, axis=0) * (1.0/m)
		w0 = w0 - learnRate * (feature.T * delta_hidden)
		b0 = b0 - learnRate * sum(delta_hidden, axis=0) * (1.0/m)

		if i % 100 == 0:
			cost=0.5*get_cost(get_predict(feature, w0, b0, w1, b1) - label)
			print('---',i,': cost = ',cost)
			# 学习率随cost下降
			learnRate=cost/5 
			if cost < delta:
				break

	return w0,b0,w1,b1

#------------------------------------------------------预测评价部分---------------------------------------------------


def get_predict(feature, w0, b0, w1, b1):
	''' 计算最终的预测
	'''
	return predict_out( predict_in( hidden_out( hidden_in(feature, w0, b0) ) , w1, b1) )

def get_cost(cost):
	m,n = cost.shape
	cost_sum= 0.0
	for i in range(m):
		for j in range(n):
			cost_sum += cost[i,j]*cost[i,j]
	return cost_sum/m

def err_rate(label, pre):
	m = label.shape[0]
	err = 0.0
	for i in range(m):
		if label[i,0]!=pre[i,0]:
			err+=1
	return err/m

#-------------------------------------------------计算函数部分----------------------------------------------------

def sig(x):
	'''sigmoid 函数
	'''	
	return 1.0/(1+exp(-x))

def partial_sig(x):
	'''sigmoid 导函数的值
	'''
	m, n = x.shape
	out = mat(zeros((m,n)))
	for i in range(m):
		for j in range(n):
			out[i,j] = sig(x[i,j]) * (1 - sig(x[i,j]))
	return out

def hidden_in(feature, w0, b0):
	'''计算隐含层的输入
	'''
	m = feature.shape[0]
	hidden_in = feature * w0
	for i in range(m):
		hidden_in[i,] += b0
	return hidden_in

def hidden_out(hidden_in):
	'''计算隐含层的输出
	'''
	hidden_output = sig(hidden_in)
	return hidden_output

def predict_in(hidden_out, w1, b1):
	'''计算输出层的输入
	'''
	m = hidden_out.shape[0]
	predict_in = hidden_out * w1
	for i in range(m):
		predict_in[i,] += b1
	return predict_in

def predict_out(predict_in):
	'''输出层的输出
	'''
	result = sig(predict_in)
	return result


#---------------------------------------------------------------------------------------------------------------

#-------------------------------------------------随机生成样本----------------------------------------------------

def generate_data():
	''' 在[-4.5,4.5]之间随机生成20000组点
	'''
	data = mat(zeros((20000, 2)))
	m = shape(data)[0]
	x = mat(random.rand(20000, 2))
	for i in range(m):
		data[i,0] = x[i,0] * 9 - 4.5
		data[i,1] = x[i,1] * 9 - 4.5
	return data

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
	M = {}
	k = 0
	for i in range(m):
		if int(Y[i,0]) not in M:
			M[int(Y[i,0])]=k	
			k+=1
	n = len(M)
	_Y=[[0]*n]*m
	_Y=mat(_Y)
	for i in range(m):
		_Y[i, M[int(Y[i,0])]]=1
	return _Y

#-------------------------------------------------生成预测图像----------------------------------------------------

def show_plot(X,Y):
	#show the hyperplane plot and distribution of all points
	c1_X1=[]
	c1_X2=[]
	c2_X1=[]
	c2_X2=[]
	n = X.shape[0]
	for i in range(n):
		if float(Y[i,0])>0.5:
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

if __name__ =="__main__":
	
	f='train.txt'
	raw_X,raw_Y=load_data(f)
	
	train_X=Normalization(raw_X)
	train_Y=transformLabel(raw_Y)

	w0, b0, w1, b1=bp_train(train_X,train_Y,10,2)
	
	test_Y = get_predict(train_X, w0, b0, w1, b1)
	test_Y = argmax(test_Y,axis=1)
	
	# 注意 此处error rate为正确率
	#	因为获得的test_Y为分类序号:[0]or[1]，而train_Y为标签所属序列[0,1]或[1,0]
	#	多分类问题中计算误差的函数需要进一步修改
	
	err=err_rate(train_Y,test_Y)
	print('accuracy = ',err)
	show_plot(raw_X,test_Y)
	
