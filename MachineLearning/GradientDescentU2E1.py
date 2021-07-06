import numpy as np
import matplotlin.pyplot as plt

# check convergence
def has_converged(theta_new, grad):
	return np.linalg.norm(grad(theta_new))/len(theta_new) < 1e-1

def GD_momentum(theta_init, eta, gamma):
	theta = [theta_init]
	v_old = np.zeros_like(theta_init)
	for it in range(100):
		v_new = gamma*v_old + eta*grad(theta[-1])
		theta_new = theta_new - v_new
		if has_converged(theta_new, grad):
			break
		theta.append(theta_new)
		v_old = v_new
	return theta
			
def sgrad(w, i, rd_id):
	true_i = re_id[i]; 			# Chọn chỉ số 
	xi = Xbar[true_i, :]		# xi - hàng thứ i
	y = y[true_i]				# yi - cột thứ i 
	a = np.dot(xi, w) - yi		# Tính xi*w - yi
	return (xi*a).reshape(2, 1)
	
def SGD(w_init, grad, eta):
    w = [w_init]
    w_last_check = w_init
    iter_check_w = 10
    N = X.shape[0]
    count = 0
    for it in range(10):
        # shuffle data 
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1 
            g = sgrad(w[-1], i, rd_id)
            w_new = w[-1] - eta*g
            w.append(w_new)
            if count%iter_check_w == 0:
                w_this_check = w_new                 
                if np.linalg.norm(w_this_check - w_last_check)/len(w_init) < 1e-3:                                    
                    return w
                w_last_check = w_this_check
    return w
