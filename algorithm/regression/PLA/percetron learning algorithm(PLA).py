'''
Perceptron Learning Algorithm(PLA)
linear(binary) classification

program running method:
	under the Python3 environment
	numpy, matplotlib packages are loaded
	we can run this program directly

input: 
	this program can make training data by itself and build a data file so there
	is no input data.
	training data consist of feature data set(two feature values) and label data.
	the range of feature value is from 0 to 100. label value is -1 or 1.


results:
	it can draw a coordinate system which contains the training data distribution 
	and a hyperplane (a line on two-dimentional coordinate system).
	and it can print the final learning coefficient set as a W vector on the CMD 
	box.

'''


from numpy import *
import random
import matplotlib.pyplot as plt

def make_data():  #make training data and save as file'training_data.txt'
	file_name='training_data.txt'
	f=open('training_data.txt','w+')
	f.seek(0,0)
	a,b,c=3,2,-200    #three parameters of underlying classification function
	for i in range(300):  #make 300 points randomly
		x=random.randint(0,100)
		y=random.randint(0,100)
		l=-1
		if x*a+y*b+c>=0:
			l=1
		#print(x,y,l)
		f.write(str(x))
		f.write('\t')
		f.write(str(y))
		f.write('\t')
		f.write(str(l))
		f.write('\n')
	f.close()
	return file_name

def load_data(file_name):  
	'''load training data from file_name
	input:  file_name(string)
	output: feature_data(matrix)
			label_data(matrix)
	'''
	f = open(file_name,'r')
	feature_data=[]
	label_data=[]
	for line in f.readlines():
		feature_tmp=[]
		label_tmp=[]
		lines=line.strip().split('\t')
		feature_tmp.append(1)
		for i in range(len(lines)-1):
			feature_tmp.append(float(lines[i]))
		label_tmp.append(float(lines[-1]))

		feature_data.append(feature_tmp)
		label_data.append(label_tmp)
	f.close()
	return mat(feature_data), mat(label_data)

def show_training_data(traindata, label):
	#show the distribution of training data on Coordinate system
    dataArr = array(traindata)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label[i]) == 1:
            xcord1.append(dataArr[i, 1])  
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    ax.scatter(xcord1, ycord1, s=10, c='#0099ff', marker='s')
    ax.scatter(xcord2, ycord2, s=10, c='#ff0066')
    plt.show()

def sigmoid(X):
    X = float(X)
    if X > 0:
        return 1
    elif X < 0:
        return -1
    else:
        return 0

def pla(train_data, train_label):
	train_data = mat(train_data)  
	train_label = mat(train_label) 
	m, n = shape(train_data)
	w = ones((n, 1))  
	while True:
		iscompleted = True
		for i in range(m):
			tag=sigmoid(dot(train_data[i], w))
			if tag == train_label[i] or tag == 0:  
				continue
			else:
				iscompleted = False
				w += (train_label[i] * train_data[i]).transpose()
		if iscompleted:
			break
	return w

def show_result(w,traindata, label):
    dataArr = array(traindata)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label[i]) == 1:
            xcord1.append(dataArr[i, 1]) 
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=10, c='#0099ff')
    ax.scatter(xcord2, ycord2, s=10, c='#ff0066')
    x = arange(0, 70, 1)
    y = (-w[0] - w[1] * x) / w[2]
    ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()

if __name__ =="__main__":
	f = make_data()
	X,Y=load_data(f)
	#print(X)  #show feature set of training data
 	#print(Y)  #show label set of training data
	#show_training_data(X,Y)   #show distribution of training data on coordinate system
	w=pla(X,Y)
	print(w)
	show_result(w,X,Y)