from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def grad(w):
	return 2*w
	
def cost(w):
	return w**2;

def numerical_grad(w, cost):
	eps = 1e-4
	g = np.zeros_like(w)
	for i in range(len(w)):
		w_p = w.copy()
		w_n = w.copy()
		w_p[i] += eps
		w_n[i] -= eps
		g[i] = (cost(w_p[i]) - cost(w_n[i]))/(2*eps)
	return g
	
def check_grad(w, cost, grad):
	w = np.random.rand(w.shape[0], w.shape[1])
	grad1 = grad(w)
	grad2 = numerical_grad(w, cost)
	return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False
	
print( 'Checking gradient...', check_grad(np.random.rand(2, 1), cost, grad))	

