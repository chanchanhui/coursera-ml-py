import numpy as np
import GradientDescent as GD
import matplotlib.pylab as plt
from scipy import linalg
#first column is the size of the house (in square feet), second column is the number of bedroom
data=np.loadtxt('ex1data2.txt',delimiter=',')
x=data[:,0:2] #m*2
y=data[:,2] #m*1
m=y.size

x_normalize,x_std,x_mean=GD.Feature_normalize(x)
alpha = 0.01
num_iters = 400

# Init theta and Run Gradient Descent
theta = np.zeros(3)
x_normalize =np.c_[np.ones(m),x_normalize]
theta1,j_history1=GD.gradientdescent_mutli(x_normalize,y,alpha,theta,num_iters)
theta,j_history2=GD.gradientdescent_mutli(x_normalize,y,0.02,theta,num_iters)
theta,j_history3=GD.gradientdescent_mutli(x_normalize,y,0.015,theta,num_iters)

#print(theta)

num_iteration_x=np.array(range(num_iters))

#Convergence of gradient descent with an appropriate learning rate
#in different learning rate

plt.figure(0)
plt.plot(num_iteration_x,j_history1,'b-',label='alpha=0.01')
plt.plot(num_iteration_x,j_history2,'r-',label='alpha=0.02')
plt.plot(num_iteration_x,j_history3,'g-',label='alpha=0.015')
plt.xlabel('iteration') #x轴
plt.ylabel('J(theta)') #y轴
plt.legend()
#plt.show()

#make predict in 1650 square feet and 3 bedrooms. 
x_pre=np.array([1650,3])
x_pre=x_pre-x_mean
x_pre=np.divide(x_pre,x_std)

x_pre_F_nor=np.concatenate((np.array([1]),x_pre),axis=0)

y_pred=theta1.dot(x_pre_F_nor)

print("make predict:1650 square feet and 3 bedrooms,by gradient descent.The price is",y_pred)



# ================ Part 3: Normal Equations ================
#compute theta by Normal Equations
def normalEqn(x, y):
   pinvof_xtx = linalg.pinv(x.T.dot(x))
   print(pinvof_xtx)
   theta=pinvof_xtx.dot(x.T).dot(y)
   return theta

x=np.c_[np.ones(m),x]
theta_3 = np.zeros(3)
theta_3=normalEqn(x,y)
print(theta_3)

#make prediction
x_pre=np.array([1650,3])
x_pre_F_nor=np.concatenate((np.array([1]),x_pre),axis=0)
y_pred=theta_3.dot(x_pre_F_nor)
print("make predict:1650 square feet and 3 bedrooms,by normal Equations.The price is",y_pred)

