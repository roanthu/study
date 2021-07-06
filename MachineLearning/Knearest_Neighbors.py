# -*- coding: utf-8 -*-
# Iris flower dataset có sẵn trong thư viện scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

# Load và hiển thị một số dữ liệu mẫu
# Các class được gán nhãn 0, 1, 2

iris = datasets.load_iris()
iris_X = iris.data					# Dữ liệu về các loại hoa lan
iris_y = iris.target				# Loại hoa tương ứng với các mẫu
print 'Number of classes: %d' %len(np.unique(iris_y))	
print 'Number of data points: %d' %len(iris_y)

X0 = iris_X[iris_y == 0, :]
print '\nSamples from class 0:\n', X0[:5, :]

X1 = iris_X[iris_y == 1, :]
print '\nSamples from class 1:\n', X1[:5, :]

X2 = iris_X[iris_y == 2, :]
print '\nSamples from class 2:\n', X2[:5, :]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

print 'Training size: %d' %len(y_train)
print 'Test size    : %d' %len(y_test)

clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)	# p = 2, tức khoảng cách được dùng là khoảng cách Euclide
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print 'Print results for 20 test data points:'
print 'Predicted labels:', y_pred[20:40]
print 'Groud truth     :', y_test[20:40]

from sklearn.metrics import accuracy_score 
print "Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred))

clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print "Accuracy of 10NN: %.2f %%" %(100*accuracy_score(y_test, y_pred))

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print "Accuracy of 10NN (1/distance ưeights): %.2f %%" %(100*accuracy_score(y_test, y_pred))
