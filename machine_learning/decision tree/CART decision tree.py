from numpy import *
import math
import random

'''
	构建CART分类树
'''

def label_uniq_cnt(data):
	#统计数据中的标签种类数
	#output: 每种样本的标签个数(字典类型)

	label_uniq_cnt={}
	for x in data:
		label = x[len(x)-1]
		if label not in label_uniq_cnt:
			label_uniq_cnt[label]=0
		label_uniq_cnt[label]+=1
	return label_uniq_cnt

def cal_gini_index(data):
	#计算数据集的gini指数
	
	total_sample = len(data)
	if len(data) == 0:
		return 0
	label_counts = label_uniq_cnt(data)
	#计算标签的种类数
	gini=0
	for label in label_counts:
		gini = gini + math.pow(label_counts[label],2)
	gini=1-float(gini)/math.pow(total_sample,2)

	return gini

class node:
	#构建分类树节点的类
	def __init__(self,fea=-1,value=None,results=None,right=None,left=None):
		self.fea=fea #切分数据集的特征的索引值
		self.value=value #设置划分的值
		self.results=results #储存叶子节点所属的标签类别
		self.right=right #左子树
		self.left=left #右子树

def split_tree(data,fea,value):
	#将数据集划分成左右子树
	#以fea特征的数值value为分界划分数据集
	set1=[]
	set2=[]
	for x in data:
		if x[fea] >= value:
			set1.append(x)
		else:
			set2.append(x)
	return (set1,set2)

def build_tree(data):
	#构建分类树
	#output: 数的根节点（携带一整棵学习好的分类树）
	if len(data) == 0:
		return node()
	#计算当前gini指数并初始化划分状态
	currentGini = cal_gini_index(data)
	bestGain = 0.0
	bestCriteria = None
	bestSets = None
	feature_num=len(data[0])-1
	
	#找到最佳划分特征值和划分点
	for fea in range(0,feature_num):
		#取得fea特征的所有可能的划分值
		feature_values = {}
		for sample in data:
			feature_values[sample[fea]] = 1

		#遍历所有可能的划分值
		for value in feature_values.keys():
			(set1,set2)=split_tree(data,fea,value)
			nowGini = float(len(set1) * cal_gini_index(set1) + len(set2) * cal_gini_index(set2)) / len(data)
			gain = currentGini - nowGini
			#判断当前划分是否更好
			if gain > bestGain and len(set1)>0 and len(set2)>0:
				bestGain = gain
				bestCriteria = (fea, value)
				bestSets = (set1,set2)

	if bestGain > 0: #若有更好的划分，则继续递归下一层分类树节点
		right = build_tree(bestSets[0])
		left = build_tree(bestSets[1])
		return node(fea=bestCriteria[0],value=bestCriteria[1],right=right,left=left)
	else: #返回当前类别标签作为最终分类标签
		return node(results=label_uniq_cnt(data))


'''用训练好的分类树进行预测
'''
def predict(sample, tree):
	'''	input: 单个样本sample
		output: 样本的所属的分类
	'''
	#递归出口，叶子节点
	if tree.results!=None:
		return tree.results
	else:
		val_sample = sample[tree.fea]
		branch = None
		if val_sample >= tree.value:
			branch = tree.right
		else:
			branch = tree.left
		return predict(sample,branch)


'''构建随机森林
'''
def choose_samples(data,k):
	'''从原始数据集中随机选择样本及特征
	input: 原始数据集data, 选择的特征数量k
	output: data_samples: 只保留K个特征值的数据
			feature(list): 保留的k个特征的索引
	'''
	m,n=shape(data)
	feature=[]
	for j in range(k):
		feature.append(random.randint(0,n-2)) # n-1列是标签
	index=[]
	for i in range(m):
		index.append(random.randint(0,m-1))
	data_samples=[]
	for i in range(m):
		data_tmp=[]
		for fea in feature:
			data_tmp.append(data[i][fea])
		data_tmp.append(data[i][-1])
		data_samples.append(data_tmp)
	return data_samples,feature

def random_forest_training(train_data,trees_num):
	'''构建随机森林
	output: trees_result(list): 每一棵树的最好划分
			trees_feature(list): 每一棵树对原始特征的选择
	'''
	trees_result = []
	trees_feature = []
	n = shape(train_data)[1]
	if n>2:
		k = int(math.log(n-1,2))+1
	else:
		k=1
	#开始构建每一棵分类树
	for i in range(trees_num):
		data_samples,feature = choose_samples(data_train,k)
		tree = build_tree(data_samples)
		trees_result.append(tree)
	trees_feature.append(feature)
	return trees_result,trees_feature



'''
训练随机森林模型
'''
def load_data(file_name):
	#导入数据
	data_train=[]
	f=open(file_name)
	for line in f.readlines():
		lines=line.strip().split(' ')
		data_tmp=[]
		for x in lines:
			data_tmp.append(float(x))
		data_train.append(data_tmp)
	f.close()
	return data_train

def split_data(data_train,feature):
	#按照选择的特征整合原始数据集
	m=data_train.shape[0]
	data=[]
	for i in xrange(m):
		data_x_tmp=[]
		for x in feature:
			data_x_tmp.append(data_train[i][x])
		data_x_tmp.append(data_train[i][-1])
		data.append(data_x_tmp)
	return data

def cal_correct_rate(data_train,final_predict):
	#计算模型预测的准确性
	m = len(final_predict)
	corr = 0.0
	for i in xrange(m):
		if data_train[i][-1] * final_predict[i]>0: 
			corr+=1   #二分类问题，符号相同则为预测正确
	return corr/m

def get_predict(trees_result,trees_feature,data_train):
	#用训练好的随机森林模型对数据集进行预测
	#output: final_predict(list): 预测数据的分类
	m_tree=len(trees_result)
	m=data_train.shape[0]
	result = []
	for i in xrange(m_tree):
		clf=trees_result[i]
		feature=trees_feature[i]
		data=split_data(data_train,feature)
		result_i=[]
		for i in xrange(m):
			result_i.append( (predict(data[i][0:-1],clf).keys())[0] )
		result.append(result_i)
	final_predict=sum(result,axit=0)
	return final_predict



if __name__ == "__main__":
	#读取学习数据
	data_train = load_data("data.txt")
	#print(data_train)
	print("load finished")
	#训练随机森林模型
	
	trees_result,trees_feature=random_forest_training(data_train,50)
	print("train finished")
	#预测
	result = get_predict(trees_result,trees_feature,data_train)
	print(result)
	#评估预测准确度
	corr_rate=cal_correct_rate(data_train,result)
	print(corr_rate)
































