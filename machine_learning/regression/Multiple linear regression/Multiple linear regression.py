'''
多元线性回归学习样例
数据：多种广告形式的投资对产品销售量的影响
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#读取数据
data = pd.read_csv("Advertising.csv")
y = data.loc[:,'sales'].as_matrix(columns=None)
y = np.array([y]).T   #将输出值y变成列表的形式后转置成一列矩阵
#print(y)
x = data.drop('sales',1)
x = x.iloc[0:,1:].as_matrix(columns=None)
#print(x)

X_train,X_test,Y_train,Y_test= train_test_split(x,y,test_size=0.3,random_state=0)
#划分数据 70%训练集 30%测试集

print(X_train)
print(Y_train)
#训练
l=LinearRegression()
l.fit(X_train,Y_train)

#每个参数的相关系数
print(l.coef_)    

#用测试集进行评分 R2系数
print(l.score(X_test,Y_test))   

#输出特定数据的预测值  [60,60,60]
#注意：必须为一行矩阵，若少一组中括号则为一列矩阵
print(l.predict([[60,60,60]]))
