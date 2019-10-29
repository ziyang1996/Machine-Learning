'''逻辑回归
训练部分+图形展示 
'''

import numpy as np
import matplotlib.pyplot as plt

def sig(x):
    ''' Sigmoid函数
    input: x(mat):feature * w
    putput: sigmoid(x)(mat):Sigmoid值
    '''
    return 1.0/(1+np.exp(-x))

def lr_train_bgd(feature, label, maxCycle, alpha):
    '''利用梯度下降法训练LR模型
    input:  feature(mat):特征值
            label(mat):标签
            maxCycle(int):最大迭代次数
            alpha(float):学习率
    output: weight(权重)
    '''
    n = np.shape(feature)[1] #特征个数
    w = np.mat(np.ones((n,1))) #初始化权重
    i = 0
    while i<=maxCycle:  #在最大迭代次数的范围内
        i += 1      #当前迭代次数
        h = sig(feature * w)    #计算Sigmoid值
        err = label - h 
        w = w +alpha * feature.T * err  #权重修正
    return w

#def error_rate():


def load_data(file_name):
    '''
    input:  file_name(string) ：训练数据的位置
    output: feature_data(mat) ：特征
            label_data(mat)   :标签
    '''
    f = open(file_name)
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
    return np.mat(feature_data), np.mat(label_data)

def save_model(file_name,w):
    '''保存最终的模型
    input:  file_name(string):模型保存的文件名
            w(mat):LR模型的权重
    '''
    m = np.shape(w)[0]
    f_w = open(file_name,"w")
    w_array=[]
    for i in range(m):
        w_array.append(str(w[i,0]))
    f_w.write(" ".join(w_array))
    f_w.close()




if __name__ == "__main__":
    # 导入Logistic Regression模型
    feature, label = load_data("data.txt")
    
    #模型训练
    w = lr_train_bgd(feature, label, 1000, 0.01)
    #保存训练出的参数
    save_model("weights",w)
    #print(w)

    

    #print(feature)
    #print(label)
    
    #将点集分类
    x_A=[]
    x_B=[]
    for i in range(feature.shape[0]):
        if int(label[i][0])==1:
            x_A.append(float(feature[i,1]))
        else:
            x_B.append((float(feature[i,1])))
    y_A=[]
    y_B=[]
    for i in range(feature.shape[0]):
        if int(label[i][0])==1:
            y_A.append(float(feature[i,2]))
        else: 
            y_B.append((float(feature[i,2])))

    ''' #企图画出超平面但是失败了T^T
    x=[]
    y=[]
    det=-4.0
    for i in range(1,100):
        x.append(det)
        det+=0.08
        y.append(1-w[2,0]*det)
    #print(x)
    #print(y)
    '''

    #可视化结果
    plt.figure(figsize=(10,12)) #设置画布大小
    plt.scatter(x_A, y_A,color = 'red')  #描出训练集对应点
    plt.scatter(x_B, y_B,color = 'blue')  #描出训练集对应点
    #plt.plot(x,y,color='black')  #企图画出的超平面
    plt.xlabel('X')  #X轴标签
    plt.ylabel('Y')  #Y轴标签
    plt.title('Train result')
    plt.show()
    