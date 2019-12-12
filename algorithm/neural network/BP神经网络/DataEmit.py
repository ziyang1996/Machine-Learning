from numpy import *
import random
import matplotlib.pyplot as plt

def generate_training_data(n):
	f=open('train.txt','w+')
	f.seek(0,0)
	data=[]
	for i in range(int(n/2)):
		x1=random.uniform(-10,10)
		x2=0.03*(x1**3) - 0.2*(x1**2) + 0.5*x1 - 12.4
		x2+=random.uniform(0,40.0)
		data.append([x1,x2,1,0])
		f.write(str(round(x1, 6)))
		f.write('\t')
		f.write(str(round(x2, 6)))
		f.write('\t')
		f.write('1')
		f.write('\n')
	for i in range(int(n/2)):
		x1=random.uniform(-10,10)
		x2=0.03*(x1**3) - 0.2*(x1**2) + 0.5*x1 - 12.4
		x2+=random.uniform(-40.0,0)
		data.append([x1,x2,0,1])
		f.write(str(round(x1, 6)))
		f.write('\t')
		f.write(str(round(x2, 6)))
		f.write('\t')
		f.write('0')
		f.write('\n')
	return mat(data)

def generate_square_data(n):
	f=open('train.txt','w+')
	f.seek(0,0)
	data=[]
	for i in range(n):
		x1=random.uniform(-10,10)
		x2=random.uniform(-10,10)
		p=sqrt(50)
		if(x1>-p and x1<p and x2>-p and x2<p):
		#if float(x1)>-5.0 and float(x1)<5.0 and float(x2)>-5.0 and float(x2)<-5.0:
			data.append([x1,x2,1,0])
			f.write(str(round(x1, 6)))
			f.write('\t')
			f.write(str(round(x2, 6)))
			f.write('\t')
			f.write('1')
			f.write('\n')
		else:
			data.append([x1,x2,0,1])
			f.write(str(round(x1, 6)))
			f.write('\t')
			f.write(str(round(x2, 6)))
			f.write('\t')
			f.write('0')
			f.write('\n')
	return mat(data)


def show_plot(X):
	#show the hyperplane plot and distribution of all points
	c1_X1=[]
	c1_X2=[]
	c2_X1=[]
	c2_X2=[]
	n = X.shape[0]
	for i in range(n):
		if int(X[i,-1])==1:
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


if __name__ =="__main__":
	#data=generate_training_data(1000)
	data=generate_square_data(1000)
	show_plot(data)
	print(data)
