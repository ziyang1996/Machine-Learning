'''逻辑回归
测试验证部分
'''

import numpy as np
import matplotlib.pyplot as plt

def sig(x):
    ''' Sigmoid函数
    input: x(mat):feature * w
    putput: sigmoid(x)(mat):Sigmoid值
    '''
    return 1.0/(1+np.exp(-x))

def load_test_data(file_name, n):
    '''导入测试数据
    input:  file_name(string):测试集的位置
            n(int):特征的个数
    output: np.mat(feature_data)(mat): 测试集的特征
    '''
    f = open(file_name)
    feature_data=[]
    for line in f.readlines():
        feature_tmp=[]
        lines = line.strip().split("\t")
        feature_tmp.append(1)
        feature_tmp.append(float(lines[0]))
        feature_tmp.append(float(lines[1]))

        feature_data.append(feature_tmp)
    f.close()
    l=len(feature_data)
    r=len(feature_data[0])
    #print(feature_data)
    return l,r,np.mat(feature_data)

def load_weight(file_name):
    '''导入LR模型
    input: file_name(string) 权重所在文件位置
    output: np.mat(w)(mat) 权重的矩阵
    '''
    f=open(file_name)
    w=[]
    for line in f.readlines():
        lines=line.split()
        w.append(float(lines[0]))
        w.append(float(lines[1]))
        w.append(float(lines[2]))
    f.close()
    return np.mat(w)

def predict(data ,w):
	'''对测试结果进行预测
	input: 	data(mat):测试数据的特征
			w(mat):模型的参数
	output:	h(mat)：最终的测试结果
	'''
	h = sig(data * w.T)
	m = np.shape(h) [0]
	for i in range(m):
		if h[i,0]<0.5:
			h[i,0]=0.0
		else:
			h[i,0]=1.0
	return h

if __name__ == "__main__":
    w = load_weight("weights")
    #print(w)

    n = np.shape(w)[1]
    l,r,testData = load_test_data("test_data.txt",n)
    print(testData)

    h = predict(testData,w)
    #print(h)
    
    x_A=[]
    y_A=[]
    x_B=[]
    y_B=[]
    for i in range(l):
        if h[i,0]==0:
            x_A.append(testData[i,1])
            y_A.append(testData[i,2])
        else:
            x_B.append(testData[i,1])
            y_B.append(testData[i,2])
    #for i in range(len(x_A)):
    #    print(x_A[i],y_A[i])
    #'''

    #可视化结果
    plt.figure(figsize=(10,12)) #设置画布大小
    plt.scatter(x_A, y_A,color = 'blue')  #描出训练集对应点
    plt.scatter(x_B, y_B,color = 'red')  #描出训练集对应点
    #plt.plot(x,y,color='black')  #企图画出的超平面
    plt.xlabel('X')  #X轴标签
    plt.ylabel('Y')  #Y轴标签
    plt.title('Train result')
    plt.show()
	#print(h)
    #'''