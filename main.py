from createmat import *
from knn import *

#解析输入的字符，如果符合2个数字的输入，则开始测试。如果是其他格式，则返回类型为string，主程序中判断是否继续
def get_cmd_pars(cmd_str):	
	#中间过度数组
	cmd_medum=[]
	#用来存放输入的数字
	pars_ret=[]
	#用来做标记
	type_ret  = 'digit'
	# 切割输入的字符串
	cmd_list = cmd_str.split(sep=' ')
	# 第一个for循环用来去掉空格
	for cl in cmd_list:
		if cl != '':
			cmd_medum.append(cl)
	# 第二个循环用来判断是否是纯数字
	for cr in cmd_medum:
		if not cr.isdigit():
			type_ret = 'string'
		else:
			#把数字存进pars_ret数组
			pars_ret.append(int(cr))
	return type_ret,pars_ret
		
#主函数
if __name__ == '__main__':
	
	train_image_file = 'train-images.idx3-ubyte'
	train_label_file = 'train-labels.idx1-ubyte'
	test_image_file = 't10k-images.idx3-ubyte'
	test_label_file = 't10k-labels.idx1-ubyte'
	
	#选择所有图片作为训练样本。
	#自己加的：第3个参数为偏移起始位置，第4个参数是训练样本数
	train_image_mat, train_label_list  = read_image_label_all_vector(train_image_file,train_label_file)

	while True:
		#如果输入的样本数量为0，判断是否退出，如果不为0，继续开始分类。
		cmd = input('请输入测试样本起始位置与数量(比如 100 200):')
		#输入的两个数字
		type_ret,par_ret = get_cmd_pars(cmd)
		offset = par_ret[0]
		amount = par_ret[1]

		#根据前面的输入偏移和数量，开始读出测试样本
		test_image_mat, test_label_list  = read_image_label_vector(test_image_file,test_label_file,offset,amount)

		#开始分类
		count = 0.0#记录错误数量
		for i in range(len(test_image_mat)):
			#利用knn算法进行分类
			class_result = knn_classify(test_image_mat[i], train_image_mat, train_label_list, 5)#计算分类结果
			print( "第 %d 张图片, KNN计算的分类结果为: %d, 实际标签为: %d" % (i+1,class_result, test_label_list[i]),end=' ')
			#判断分类结果是发和标签一致
			if (class_result != test_label_list[i]):
				print(' 分类错误！',end = ' ')
				count += 1.0
			#打印错误率
			print('实时错误率反馈：%2.2f%%' % (100.0*count/(i+0.01)))

		print( "\n错误总数: %d" % count)
		print( "正确率: %2.2f%%" % (100-100.0*count/len(test_image_mat)))
		print(f"测试数据集标签:\n{test_label_list}")




