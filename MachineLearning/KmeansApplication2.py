# -*- coding: utf-8 -*-
# Khai báo các thư viện

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

img = mpimg.imread('girl3.jpg')		# Load ảnh
plt.imshow(img)						# Hiển thị ảnh trên các trục tọa độ
imgplot = plt.imshow(img)			# Trả về AxesImage
plt.axis('off')						# Ẩn trục tọa độ
plt.show()

X = img.reshape((img.shape[0]*img.shape[1], img.shape[2])) 	# Biến đổi bức ảnh thành một ma trận có mỗi hàng là 1 điểm ảnh với 3 giá trị màu

for K in [2, 5, 10, 15, 20]:								# Sử dụng số lượng cluster khác nhau
    kmeans = KMeans(n_clusters=K).fit(X)
    label = kmeans.predict(X)

    img4 = np.zeros_like(X)									# Trả về một mảng toàn giá trị 0 có kích cỡ và kiểu tương tự X
    # Thay thế mỗi điểm ảnh bằng center của nó
    for k in range(K):
        img4[label == k] = kmeans.cluster_centers_[k]
    # Reshape và hiển thị ảnh đầu ra
    img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
    plt.imshow(img5, interpolation='nearest')
    plt.axis('off')
    plt.show()

