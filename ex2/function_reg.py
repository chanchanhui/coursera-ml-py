import numpy as np
import	matplotlib.pylab as plt
def loadtxt():
	data_np=np.loadtxt("ex2data2.txt", delimiter=',')
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

	#plt.show()
	
def sigmoid(x):
    g = 1/(1+np.exp(-1*x))
    return g

def costFuncReg(theta,X ,y,Lambda):
	m=np.size(y,0)
	h=sigmoid(X.dot(theta)) # m*2 2*1=m*1
	p1=-(y.dot(np.log(h))+(1-y).dot(np.log(1-h)))/m
	p2=(Lambda/(2*m)) * theta[1:].dot(theta[1:])#np.sum(np.square(theta)[1:])
	# p2=Lambda/(2*m)*theta[1:].dot(theta[1:])
	print(p1) 
	print(p2)
	return p1+p2
	# m = np.size(y, 0)
	# h = sigmoid(X.dot(theta))
	# j=-1/m*(y.dot(np.log(h))+(1-y).dot(np.log(1-h)))+Lambda/(2*m)*theta[1:].dot(theta[1:])
	# return j

def gradFuncReg(theta, X, y, lam):
	m=np.size(y,0)
	grad=np.zeros(np.size(theta,0))
	h=sigmoid(X.dot(theta)) #m*n n*1
	#y=y.reshape(np.shape(h))
	temp=1/m * (X[:,0].dot(h-y) )#1*m m*1
	grad[0]=temp
	grad[1:]=1/m * (X[:,1:].T.dot(h-y) )+(lam/m)*theta[1:]#(n-1),m *(m,)  +(n-1,)
	return grad

def mapFeature(x1,x2):
	degree = 6
	result=np.ones((x1.size))
	#print(result)
	#x1 = x1.reshape((x1.size, 1))
	#x2 = x2.reshape((x2.size, 1))

	for i in range(1,degree+1):
		for j in range(0,i+1):
			result=np.c_[result,(x1**(i-j))*(x2 **j) ]
	return result


def plotDecisonBoundary(theta,X,y):
	plotData()
	zi=np.linspace(-1.5,1.5,50)
	zj=np.linspace(-1.5,1.5,50)
	z=np.ones((zi.size,zj.size))
	print(np.shape(z))
	for i in range(0,zi.size):
		for j in range(0,zj.size):
			z[i][j]=np.dot(mapFeature(zi[i],zj[j]),theta)

	plt.contour(zi,zj,z, levels=[0], colors='black', label='Decision Boundary')
	plt.show()

def prediction(test_data,theta):
	val=np.dot(mapFeature(test_data[:,0],test_data[:,1]),theta) #(m,1)
	val[val>0]=1
	val[val<0]=0
	return val