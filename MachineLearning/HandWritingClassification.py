# -*- coding: utf-8 -*-

import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from display_network import *

mntrain = MNIST('/home/hanlong/Documents/MachineLearning/MNIST')
mntrain.load_training()
Xtrain_all = np.asarray(mntrain.train_images)			# Trả về mảng (N, 784) với N là số điểm dữ liệu
ytrain_all = np.array(mntrain.train_labels.tolist())	

mntest = MNIST('/home/hanlong/Documents/MachineLearning/MNIST')
mntest.load_testing()
Xtest_all = np.asarray(mntest.test_images)
ytest_all = np.array(mntest.test_labels.tolist())

def extract_data(X, y, classes):
	"""
	X: Mảng numpy, ma trận kích thước (N, d) với d là số chiều của dữ liệu
	y: Mảng numpy, kích thước (N, 1)
	cls: 2 list các nhãn. Ví dụ: 
		cls = [[1, 4, 7], [5, 6, 8]]
	Trả về:
		X: Dữ liệu đã trích chọn
		y: Nhãn đã trích chọn 
		(0 và 1, tương ứng với 2 list trong cls)
	"""
	y_res_id = np.array([])
	for i in cls[0]:
		y_res_id = np.hstack((y_res_id, np.where(y==i)[0])) 
	n0 = len(y_res_id)
	
	for i in cls[1]:
		y_res_id = np.hstack((y_res_id, np.where(y==i)[0]))
	n1 = len(y_res_id) - n0
	
	y_res_id = y_re_id.astype(int)
