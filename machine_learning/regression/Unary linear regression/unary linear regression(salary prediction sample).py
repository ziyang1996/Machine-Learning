'''
一元线性回归学习样例
数据：工作时间（年）对工资的影响
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#载入数据并对数据预处理：
dataset = pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,0].values    # 根据行号读取第一个数据作为样本特征集
Y=dataset.iloc[:,-1].values   # 根据行号读取最后一个数据作为样本标签集
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.3,random_state=0)
X_train = X_train.reshape(-1,1)   #  reshape(-1,1):将矩阵转换成一列矩阵
Y_train = Y_train.reshape(-1,1)
'''
train_test_split( train_data , train_target , test_size=0.3 , random_state=0)
train_data：被划分的样本特征集
train_target：被划分的样本标签
test_size：如果是浮点数，在0-1之间，表示样本占比；如果是整数的话就是样本的数量
random_state：是随机数的种子。填0或不填，每次随机结果不同；其他则是固定随机数
'''

#创建回归器,拟合训练集并作出测试集的预测
regresor = LinearRegression()
regresor.fit(X_train,Y_train)  #拟合
#根据模型得出的对应测试集的预测Y值
predict=regresor.predict(X_test.reshape(-1,1))




#可视化结果
plt.figure(figsize=(10,12)) #设置画布大小
figure = plt.subplot(211)  #将画布分成2行1列，当前位于第一块子画板
plt.scatter(X_train, Y_train,color = 'red')  #描出训练集对应点
plt.plot(X_test,predict,color='black')  #画出预测的模型曲线
plt.xlabel('YearsExperience')  #X轴标签
plt.ylabel('Salary')           #Y轴标签
plt.title('Train set')

#将模型应用于测试集，检验拟合结果
figure=plt.subplot(212)   #当前位于第二块子画板
plt.scatter(X_test, Y_test,color = 'red')    #描出测试集对应点
plt.plot(X_test,predict,color='black')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Test set')

#显示画布
plt.show()

#模型评估
from sklearn.metrics import r2_score
print('r2_score : ' + str(r2_score(Y_test,predict)))
#R²分数: 相关指数，确定系数。最佳分数为1.0，可以为负数（因为模型可能会更糟)






