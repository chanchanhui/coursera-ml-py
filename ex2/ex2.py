from function import plotData,loadtxt,costFunction,gradFunction,plotDecisonBoundary,predict
import numpy as np
import scipy.optimize as op
#drawing the scatter plot
plotData()
data_np=loadtxt()

X=data_np[:,0:2]
Y=data_np[:,2]
m,n=np.shape(X)
init_theta=np.zeros((n+1,))
X=np.c_[np.ones((m,)),X]
print(init_theta)

cost = costFunction(init_theta, X, Y)
grad = gradFunction(init_theta, X, Y)
print('Cost at initial theta (zeros): ', cost)
print('Gradient at initial theta (zeros): ', grad)
_ = input('Press [Enter] to continue.')

#minimize the castFunction by scipy.optimize.minimize
result = op.minimize(costFunction, x0=init_theta, method='BFGS', jac=gradFunction, args=(X, Y))
theta=result.x
print(theta)


plotDecisonBoundary(theta,X,Y)
#predict by training data
p=predict(theta, X)
#print(data_np[:,2])
one_sum=np.sum(p==Y)
Accuracy=one_sum/np.size(X,0)
print('Accuracy:',Accuracy)