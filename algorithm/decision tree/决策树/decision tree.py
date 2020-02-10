
'''
decision tree 决策树

混乱度评价标准： 	基尼指数（Gini index）
				Gini = 1-p1^2-p2^2

注意： feature值必须为int,否则无法遍历存入字典
'''



from numpy import *
import math
import random
import matplotlib.pyplot as plt

class node:
	#构建分类树节点的类
	def __init__(self,fea=-1,value=None,results=None,right=None,left=None):
		self.fea=fea #切分数据集的特征的索引值
		self.value=value #设置划分的值
		self.results=results #储存叶子节点所属的标签类别
		self.right=right #左子树
		self.left=left #右子树


#------------------------决策树划分----------------------------

def build_tree(data):
	#构建分类树
	'''
	input:	data(list): 训练样本
	output: node: 数的根节点（携带一整棵学习好的分类树）
	'''
	if len(data) == 0:
		return node()

	#计算当前gini指数并初始化划分状态
	currentGini = cal_gini_index(data)
	bestGain = 0.0
	bestCriteria = None
	bestSets = None

	m,n=data.shape
	print('current Gini: ',currentGini)
	#找到最佳划分特征值和划分点
	for fea in range(0,n-1):
		#取得fea特征的所有可能的划分值
		feature_values = {}

		for i in range(m):
			feature_values[data[i,fea]] = 1
		#print(feature_values.keys())
		#遍历所有可能的划分值
		for value in feature_values.keys():
			
			set1,set2=split_tree(data,fea,value)
			#print(data.shape,set1.shape,set2.shape)
			nowGini = float(len(set1) * cal_gini_index(set1) + len(set2) * cal_gini_index(set2)) / len(data)
			gain = currentGini - nowGini
			#判断当前划分是否更好

			if gain > bestGain and set1.shape[0]>0 and set2.shape[0]>0:
				bestGain = gain
				bestCriteria = (fea, value)
				bestSets = (set1,set2)

	if bestGain > 0: #若有更好的划分，则继续递归下一层分类树节点
		right = build_tree(bestSets[0])
		left = build_tree(bestSets[1])
		return node(fea=bestCriteria[0],value=bestCriteria[1],right=right,left=left)
	else: #返回当前类别标签作为最终分类标签
		return node(results=label_uniq_cnt(data))


def split_tree(data,fea,value):
	#将数据集划分成左右子树
	#以fea特征的数值value为分界划分数据集
	'''
	input:	data(list):	数据集
			fea(int): 待分割的特征序号
			value(float): thresh值
	output:	set1,set2(tuple): 分割后的左右子树
	'''
	set1=[]
	set2=[]
	m,n=data.shape
	for x in range(m):
		s=[]
		for j in range(n):
			s.append(int(data[x,j]))
		if data[x,fea] >= value:
			set1.append(s)
		else:
			set2.append(s)
	return mat(set1),mat(set2)



#------------------------计算基尼指数--------------------------

def cal_gini_index(data):
	#计算数据集的gini指数
	
	m,n = data.shape #样本的总数
	if n <= 0 or m <= 0:
		return 0.0

	label_counts = label_uniq_cnt(data)
	#计算标签的种类数
	gini=0
	for label in label_counts:
		gini = gini + math.pow(label_counts[label],2)
	gini=1-float(gini)/math.pow(m,2)

	return gini


def label_uniq_cnt(data):
	#统计数据中的标签种类数
	#output: 每种样本的标签个数(字典类型)

	label_uniq_cnt={}
	m,n=data.shape

	for x in range(m):
		#print(x,m,data[x,:])
		label = int(data[x,-1])
		if label not in label_uniq_cnt:
			label_uniq_cnt[label] = 0
		label_uniq_cnt[label] = label_uniq_cnt[label] + 1
	return label_uniq_cnt



#---------------------------预测分类结果-----------------------
def predict(sample, tree):
	'''	input: 单个样本sample
		output: 样本的所属的分类
	'''
	#递归出口，叶子节点
	if tree.results!=None:
		return tree.results
	else:
		val_sample = sample[0,tree.fea]
		branch = None
		if val_sample >= tree.value:
			branch = tree.right
		else:
			branch = tree.left
		return predict(sample,branch)

def getPredict(tree, data):
	pre = []
	m = data.shape[0]
	for i in range(m):
		x=predict(data[i,:-1],tree)
		pre.append(max(x, key=x.get))
	return mat(pre).T


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
			_X.append(int(float(ls[i])*100))
		_X.append(int(ls[-1]))
		#_Y.append(int(ls[-1]))
		X.append(_X)
		#Y.append(_Y)
	f.close()
	return mat(X)

#-------------------------------------------------生成预测图像----------------------------------------------------

def show_plot(X,Y):
	#show the hyperplane plot and distribution of all points
	c1_X1=[]
	c1_X2=[]
	c2_X1=[]
	c2_X2=[]
	n = X.shape[0]
	for i in range(n):
		if int(Y[i,0])==1:
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



#-----------------------------------------------------主函数----------------------------------------------------
if __name__ == "__main__":
	f='train.txt'
	data = load_data(f)
	#print(data)
	#print(raw_Y.shape)

	decisionTree = build_tree(data[:100,:])

	#print(predict(data,decisionTree).keys()[0])
	#print(data.shape)
	predict=getPredict(decisionTree,data)
	print(predict)
	show_plot(data,predict)
	
	