import numpy as np
import matplotlib.pyplot as plt
import math

def loadtxt():
	data_np=np.loadtxt("ex2data1.txt", delimiter=',')
	return data_np
def plotData():
	data_np=loadtxt()
	data_zero = data_np[data_np[:,2] == 0 ]
	data_one =data_np [data_np[:,2] == 1]
	#Scatter plot:scatter()
	plt.scatter(data_zero[:,0],data_zero[:,1],marker="o",label="Not Adimtted",color='r', s=30)
	plt.scatter(data_one[:,0],data_one[:,1],marker="+",label="Adimtted", color='g', s=30)
	plt.legend()
	plt.xlabel('Exam 1 Sore')
	plt.ylabel('Exam 2 Sore')

	plt.show()

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

#costFunction for logistic regression
def costFunction(initial_theta, X, y):
	m=y.size
	h=sigmoid(X.dot(initial_theta)) # m*2 2*1=m*1
	#print(h ,np.sum(1-h < 1e-10))
	if np.sum(1-h < 1e-10) != 0:
		return np.inf
	j=-(y.dot(np.log(h))+(1-y).dot(np.log(1-h)))/m
	return j

def gradFunction(initial_theta, X, y):
	m=np.size(y,0)
	# y=y.reshape(m,1)
	h=sigmoid(X.dot(initial_theta))
	temp=h-y
	grad=1/m * (X.T.dot(temp) )#2*m m*1
	#grad = 1 / m * (X.T.dot(sigmoid(X.dot(initial_theta)) - y))
	return grad


def plotDecisonBoundary(theta,X,y):
	data_np=loadtxt()
	data_zero = data_np[data_np[:,2] == 0 ]
	data_one =data_np [data_np[:,2] == 1]
	#Scatter plot:scatter()
	plt.scatter(data_zero[:,0],data_zero[:,1],marker="o",label="Not Adimtted",color='r', s=30)
	plt.scatter(data_one[:,0],data_one[:,1],marker="+",label="Adimtted", color='g', s=30)
	#decision bounary: θ'X = 0,θ1 + θ2*x2 + θ3*x3 = 0
	#the line is straight，two point is enough
	plotx=np.array([np.min(X[:,1])-2,np.max(X[:,1])+2])
	ploty=-1/theta[2]*(theta[1]*plotx+theta[0])
	plt.plot(plotx,ploty)
	plt.legend()
	plt.xlabel('Exam 1 Sore')
	plt.ylabel('Exam 2 Sore')

	plt.show()

def predict(theta,X):
	m=np.size(X,0)
	p=np.zeros((m,))
	Vals=X.dot(theta)
	pos = np.where(X.dot(theta) >= 0)
	# neg = np.where(X.dot(theta) < 0)
	Vals[Vals>0] = 1.0
	Vals[Vals<0] = 0.0
	# p[p>0]=1
	# p[p<0]=0
	return Vals
