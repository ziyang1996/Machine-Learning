import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def PolyMatrix(x):#将一元一次矩阵转化为多项式矩阵
	M=[]
	for i in range(len(x)):
		#常数项，x^2,x^3,x^4 (可扩展，可能发生过拟合)
		a=[1,x[i],x[i]**2,x[i]**3]
		M.append(a)
	M=np.matrix(M)
	return M

'''构造初始数据
X: 一元一次矩阵，单一属性值
Y：标签值
'''
X=[]
k=0
for i in range(0,100):
	X.append(k+random.random()/20)
	k=k+0.1
Y=[]
for i in X:
	Y.append(2*(i**3)- 20*(i**2) + 9*i+500+random.random()*10)
Y=np.array(Y).reshape(-1,1)

#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
#将一次矩阵转化为多项式矩阵
PM=PolyMatrix(X)

l=LinearRegression()
l.fit(PM,Y)
print(l.coef_)

#构造学习结果曲线
Z=[]
k=0
for i in range(0,100):
	Z.append(k)
	k=k+0.1
PZ=PolyMatrix(Z)

predict=l.predict(PZ)
#predict=predict.reshape(1,-1)
ZZ=[]
for i in range(100):
	ZZ.append(predict[i,0])

Z=np.array(Z).reshape(-1,1)
ZZ=np.array(ZZ).reshape(-1,1)


#可视化结果
plt.figure(figsize=(10,12)) #设置画布大小
figure = plt.subplot(211)  #将画布分成2行1列，当前位于第一块子画板
plt.scatter(X, Y,color = 'red')  #描出训练集对应点
plt.plot(Z,ZZ,color='black')  #画出预测的模型曲线
plt.xlabel('X')  #X轴标签
plt.ylabel('Y')           #Y轴标签
plt.title('Train set')
plt.show()

