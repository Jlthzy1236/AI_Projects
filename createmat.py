import os
from matplotlib import pyplot as plt  
import numpy as np

#图片的大小
IMAGE_ROW = 28
IMAGE_COL = 28
IMAGE_SIZE = 28*28

print("👇载入数据中，从1数到10，精彩即将开始👇")
#获取文件头dimension数据,入参：filename, 文件名称  返回：返回文件头的dimension数据
def read_head(filename):
	#print('读取文件头：',os.path.basename(filename))
	dimension = []
	with open(filename,'rb') as pf:
		#获取magic number
		data = pf.read(4)#读出第1个4字节
		magic_num = int.from_bytes(data,byteorder='big')#bytes数据大尾端模式转换为int型
		#print('magcinum: ', hex(magic_num))
		
		#获取dimension的长度，由magic number的最后一个字节确定
		dimension_cnt = magic_num & 0xff
		
		#获取dimension数据，
		#dimension[0]表示图片的个数,如果是3维数据,dimension[1][2]分别表示其行/列数值
		for i in range(dimension_cnt):
			data = pf.read(4)
			dms = int.from_bytes(data,byteorder='big')
			#print('dimension %d: %d'%(i,dms))
			dimension.append(dms)
	
	#print(dimension)
	return dimension

#功能:获取样本数量 入参：dimension, read_head()返回的维度  返回：样本数量
def get_sample_count(dimension):
	return dimension[0]


#功能:文件头的长度为4字节的magic num+dimension的个数*4
#入参：dimension, read_head()返回的维度
#返回：文件头的长度

def get_head_length(dimension):
	return 4*len(dimension)+4
	


#功能：读出文件中的第n张图片,mnist单张图片的数据为28*28个字节
#入参：filename, 样本图片的文件名称 head_len, 文件头长度 offset, 偏移位置或者图片的索引号，从第offset张图片开始的位置
#返回：image,图片的numpy数组，单张图片格式

def read_image(filename,head_len,offset):
	image = np.zeros((IMAGE_ROW,IMAGE_COL),dtype=np.uint8)#创建一个28x28的array，数据类型为uint8
	
	with open(filename,'rb') as pf:
		#magic_num的长度为4，dimension_cnt单个长度为4,前面的number个长度为28*28*offset	
		pf.seek(head_len+IMAGE_SIZE*offset) 
		
		for row in range(IMAGE_ROW):#处理28行数据，
			for col in range(IMAGE_COL):#处理28列数据
				data = pf.read(1)#单个字节读出数据
				pix = int.from_bytes(data,byteorder='big')#由byte转换为int类型，
				#简单滤波，如果该位置的数值大于指定值，则表示该像素为1.因为array已经初始化为0了，如果小于该指定值，不需要变化
				if pix >10:image[row][col] = 1
		print(image)
	
	return image

#功能：读出文件中的第n张图片对应的label
#入参：filename, 样本标签的文件名称 head_len, 文件头长度 offset, 偏移位置或者标签的索引号，从第offset个标签开始的位置
#返回：label,标签list

def read_label(filename,head_len,offset):
	label = None
	
	with open(filename,'rb') as pf:
		#pf 指向label的第number个数据,magic_num的长度为4，dimension_cnt单个长度为4
		pf.seek(head_len+offset) 
		data = pf.read(1)
		label = int.from_bytes(data,byteorder='big')#由byte转换为int类型，	
	print('读到的标签值：',label)
	return label


#功能：读出文件中的第offset张图片开始的amount张图片,mnist单张图片的数据为28*28个字节
#入参：filename, 样本图片的文件名称 head_len, 文件头长度 offset, 偏移位置，从第offset张图片开始的位置 amount, 要返回的图像数量
#返回：image_mat,图片的numpy数据

def read_image_vector(filename,head_len,offset,amount):
	image_mat=np.zeros((amount,IMAGE_SIZE),dtype=np.uint8)
	
	with open(filename,'rb') as pf:
		#magic_num的长度为4，dimension_cnt单个长度为4,前面的number个长度为28*28*offset	
		pf.seek(head_len+IMAGE_SIZE*offset) 
		
		for ind in range(amount):
			image = np.zeros((1,IMAGE_SIZE),dtype=np.uint8)#创建一个1，28x28的array，数据类型为uint8
			for row in range(IMAGE_SIZE):#处理28行数据，
				data = pf.read(1)#单个读出数据
				pix = int.from_bytes(data,byteorder='big')#由byte转换为int类型，
				#简单滤波，如果该位置的数值大于指定值，则表示该像素为1.因为array已经初始化为0了，如果小于该指定值，不需要变化
				if pix >10:image[0][row] = 1
			image_mat[ind,:]=image
			print('read_image_vector：当前进度%0.2f%%'%(ind*100.0/amount),end='\r')
		print()
		#print(image)
	
	return image_mat		

#功能：读出文件中的第n张图片开始的amout个的label
#入参：filename, 样本标签的文件名称 head_len, 文件头长度 offset, 偏移位置，从第offset张图片开始的位置 amount, 要返回的图像数量
#返回：label_list,标签list

def read_label_vector(filename,head_len,offset,amount):
	label_list=[]
	
	with open(filename,'rb') as pf:
		#pf 指向label的第number个数据,magic_num的长度为4，dimension_cnt单个长度为4
		pf.seek(head_len+offset) 
		
		for ind in range(amount):
			data = pf.read(1)
			label = int.from_bytes(data,byteorder='big')#由byte转换为int类型，	
			label_list.append(label)
			print('read_label_vector：当前进度%0.2f%%'%(ind*100.0/amount),end='\r')
		print()
	
	return label_list


#功能：从文件中读offset起始位置开始读出amout个image和label。
#入参：image_file, 图片文件名称 label_file, 标签文件名称 offset, 偏移位置，从第offset张图片开始的位置 amount, 要返回的样本数量
#返回：image_mat, 图片的numpy数组 label_list,标签list

def read_image_label_vector(image_file,label_file,offset,amount):
	
	image_dim = read_head(image_file)
	label_dim = read_head(label_file)
	
	#判断样本中的image和label是否一致
	image_amount = get_sample_count(image_dim)
	label_amount = get_sample_count(label_dim)
	if image_amount != label_amount:
		print('Error:训练集image和label数量不相等')
		return None,None
	
	if offset+amount > image_amount:
		print('Error:请求的数据超出样本数量')
		return None,None
	
	#获取样本image和label的头文件长度
	image_head_len = get_head_length(image_dim)
	label_head_len = get_head_length(label_dim)
	
	#得到image和label的向量
	image_mat = read_image_vector(image_file,image_head_len,offset,amount)
	label_list = read_label_vector(label_file,label_head_len,offset,amount)
	
	return image_mat,label_list


#功能：从文件中读取所有的image和label。
#入参：image_file, 图片文件名称 label_file, 标签文件名称
#返回：image_mat, 图片的numpy数组 label_list,标签list


def read_image_label_all_vector(image_file,label_file):
	
	image_dim = read_head(image_file)
	label_dim = read_head(label_file)
	
	#判断样本中的image和label是否一致
	image_amout = get_sample_count(image_dim)
	label_amout = get_sample_count(label_dim)
	if image_amout != label_amout:
		print('Error:训练集image和label数量不相等')
		return None,None
	
	#获取样本image和label的头文件长度
	image_head_len = get_head_length(image_dim)
	label_head_len = get_head_length(label_dim)
	
	#得到image和label的向量
	image_mat = read_image_vector(image_file,image_head_len,0,image_amout)
	label_list = read_label_vector(label_file,label_head_len,0,label_amout)
	
	return image_mat,label_list

	




