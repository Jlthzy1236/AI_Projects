from numpy import *
import numpy as np
import operator

def knn_classify(test_data, train_dataset, train_label, k):
	# 获得训练样本个数
	train_dataset_amount = train_dataset.shape[0]
	#将输入test_data变成了和train_dataset行列数一样的矩阵
	test_rep_mat =  np.tile(test_data, (train_dataset_amount,1))

	#求差,将平方后的数据相加，sum(axis=1)是将一个矩阵的每一行向量内的数据相加，得到一个list，list的元素个数和行数一样;
	#开平方，得到欧式距离
	distance = (np.sum((test_rep_mat - train_dataset)**2, axis=1))**0.5
	
	#argsort 将元素从小到大排列，得到这个数组元素在distance中的index(索引)，dist_index元素内容是distance的索引
	dist_index = distance.argsort()
	#新建一个字典
	class_count={}		  
	for i in range(k):
		#找距离最近的三个点的标签
		label = train_label[dist_index[i]]
		#如果属于某个类，在该类的基础上加1，相当于增加其权重，
		# 如果不是某个类则新建字典的一个key并且等于1(本来是为0的，后面加了个1)
		class_count[label] = class_count.get(label,0) + 1
	#降序排列,item是将字典中每对key和value组成一个元组
	#operator.itemgetter获取key对象第一个域的值
	class_count_list = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
	#print('排序后的分类结果：',class_count_list)
	#print(dist_index)
	return class_count_list[0][0]

#写一个主函数来测试下
if __name__ == '__main__':
	train_data_set=array([[2.2,1.4],\
		[2.4,2.3],\
		[1.1,3.4],\
		[8.3,7.3],\
		[9.2,8.3],\
		[10.2,11.1],\
		[11.2,9.3]])
	train_label = ['A','A','A','B','B','B','B']
	test_data = [4.6,3.4]
	print('分类结果为：',knn_classify(test_data,train_data_set,train_label,3))
	
