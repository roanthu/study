# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(2)

means = [[2, 2], [4, 2]]
cov  = [[.3, .2], [.2, .3]]
#  | 
#  2-----*-----*
#  |	 |     |
#  1     |     |
#  |     |     |
#__0__1__2__3__4

N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis = 1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)

# Xbar =  
#	 1		1 	...	 1
#   x01	   x02	... x0N
#   ...    ...  ... ...
#   xd1    xd2  ... xdN

X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)

def h(w, x):
	return np.sign(np.dot(w.T, x))
	
def has_converged(X, y, w):
	return np.array_equal(h(w, X), y)
	
def perceptron(X, y, w_init):
	w = [w_init]
	N = X.shape[1]		# Số lượng điểm dữ liệu
	d = X.shape[0]		# Số chiều của 1 điểm dữ liệu
	mis_points = []		# Tập các điểm bị phân bố sai
	
	while True:
		mix_id = np.random.permutation(N)	# các chỉ số là ngẫu nhiên
		for i in range(N): 
			xi = X[:, mix_id[i]].reshape(d, 1)
			yi = y[0, mix_id[i]]
			if h(w[-1], xi)[0] != yi: 		# Nếu xi nằm ở sai lớp
				mis_points.append(mix_id[i])
				w_new = w[-1] + yi*xi
				w.append(w_new)
		
		if has_converged(X, y, w[-1]):
			break
			
	return (w, mis_points)
    
d = X.shape[0]
w_init = np.random.rand(d, 1)
(w, m) = perceptron(X, y, w_init)
