import numpy as np
import matplotlib.pyplot as plt
import sklearn import svm

# XOR datasets and targets
X = np.c_[(0, 0), 
		  (1, 1), 
		  #-----
		  (1, 0),
		  (0, 1)].T
Y = [0]*2 + [1]*2

fignum = 1

for kernel in ('sigmoid', 'poly', 'rbf'):
	clf = svm.SVC(kernel = kernel, gamma = 4, coef = 0)
	clf.fit(X, Y)
	with PdfPages(kernel + '2.pdf') as pdf:
		fig, ax = plt.subplots()
		plt.figure(fignum, figsize=(4, 3))
		plt.clf()
		
		plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s = 80, facecolors = 'None')
		plt.plot(X[:2, 0], X[:2, 1], 'ro', marketsize = 8)
		plt.plot(X[2:,0], X[2:, 1], 'bs', marketsize = 8)

plt.axis('tight')
        x_min, x_max = -2, 3
        y_min, y_max = -2, 3
        
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.figure(fignum, figsize=(4, 3))
        CS = plt.contourf(XX, YY, np.sign(Z), 200, cmap='jet', alpha = .2)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                    levels=[-.5, 0, .5])
        plt.title(kernel, fontsize = 15)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.xticks(())
        plt.yticks(())
        fignum = fignum + 1
        pdf.savefig()
plt.show()
