# -*- coding: utf-8 -*-

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(11)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)	
# X0 là tập N điểm có hoành độ và tung độ là các số ngẫu nhiên phân bố theo phân phối chuẩn 
# với kỳ vọng, phương sai) lần lượt là (2, 1) (ứng với hoành độ) và (2,1) (ứng với tung độ).
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T


def kmeans_display(X, label):
	K = np.amax(label) + 1
	X0 = X[label == 0, :]
	X1 = X[label == 1, :]
	X2 = X[label == 2, :]

	plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
	plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
	plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)
	
	plt.axis('equal')
	plt.plot()
	plt.show()

kmeans_display(X, original_label)

def kmeans_init_centers(X, k):
	# Chọn ngẫu nhiên k hàng trong X làm các center
	# replace = False nghĩa là không được lặp lại
	return X[np.random.choice(X.shape[0], k, replace = False)]
	
def kmeans_assign_labels(X, centers):
	# Tính khoảng cách từ các điểm dữ liệu tới các centers
	# cdist([x1, x2], [y1, y2]) = [[dx1y1, dx1y2], [dx2y1, dx2y2]]
	D = cdist(X, centers)
	
	# Trả về chỉ số của center gần nhất
	# Tìm giá trị nhỏ nhất theo hàng => Đưa ra chỉ số của phần tử nhỏ nhất = (đồng thời) chỉ số của cụm mà nó được phân vào
	return np.argmin(D, axis = 1)
	
def kmeans_update_centers(X, labels, K):
	centers = np.zeros((K, X.shape[1]))
	for k in range(K): 
		# Thu thập mọi điểm được gán vào cụm k
		Xk = X[labels == k, :]
		# Tính trung bình, phần tử thứ k có tọa độ bằng trung bình tọa độ của các điểm dữ liệu
		centers[k, :] = np.mean(Xk, axis = 0)
	return centers 
	
def has_converged(centers, new_centers):
	# Trả về True nếu tập center giữ nguyên
	return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))
	
def kmeans(X, K):
	centers = [kmeans_init_centers(X, K)]
	# Ở đây là [kmeans_init_centers(X, K)] chứ không phải kmeans_init_centers(X, K)
	labels = []
	it = 0
	while True:
		labels.append(kmeans_assign_labels(X, centers[-1]))
		# centers[-1] là phần tử cuối cùng của list centers (một kiểu dữ liệu trong python), đây là các center được cập nhật lần cuối.
		new_centers = kmeans_update_centers(X, labels[-1], K)
		if has_converged(centers[-1], new_centers):
			break
			centers.append(new_centers)
			it += 1
	return (centers, labels, it)
		
(centers, labels, it) = kmeans(X, K)

print 'Centers found by our algorithm:'
print centers[-1]
kmeans_display(X, labels[-1])

kmeans = KMeans(n_clusters = 3, random_state = 0).fit(X)
print 'Centers found by scikit-learn:'
print kmeans.cluster_centers_
pred_label = kmeans.predict(X)
kmeans_display(X, pred_label)

		
