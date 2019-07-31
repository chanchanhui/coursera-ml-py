import numpy as np
def gradientdescent(X,Y,theta,alpha,iterations):
	m = Y.size
	J_history = np.zeros(iterations)

	for i in range(0,iterations):
		Jder_the0=(np.sum(np.dot(X,theta)-Y))/m #theta[0] X(m*2) Y(m,)  theta(2,)
		theta[0]=theta[0] - alpha * Jder_the0
		Jder_the1=(np.sum( np.dot( (np.dot(X,theta)-Y),X[:,1]))) / m
		theta[1]=theta[1] - alpha * Jder_the1
		#print(theta)
		# Save the cost every iteration
		J_history[i] = compute_cost(X, Y, theta)
		#print(J_history[i] )
	return theta,J_history

def gradientdescent_mutli(x,y,alpha,theta,num_iters):
	m = np.size(y, 0)
	J_history = np.zeros(num_iters)
	for i in range(num_iters):
		inner=np.dot(x,theta)-y#x:m*3,theta:3*1 y:m*1 inner:m*1
		theta1=alpha*(x.T.dot(inner))/m #theta0,theta1,2同时更新
		theta=theta-theta1
		J_history[i] = compute_cost(x, y, theta)
	return theta,J_history

def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size
    cost = 0

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set the variable "cost" to the correct value.
    inner=np.square(np.dot(X,theta)-y)#
    cost=np.sum(inner) / (2*m)
    # ==========================================================

    return cost
def Feature_normalize(X):
	x_mean = np.mean(X, axis=0)
	x_std = np.std(X, axis=0)
	x_norm = np.divide(X-x_mean, x_std)
	return x_norm,x_std,x_mean
