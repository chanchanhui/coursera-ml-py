import numpy as np
import matplotlib.pyplot as plt
import GradientDescent as GD
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.colors import LogNorm
# ===================== Part 1: Plotting =====================
data=np.loadtxt('ex1data1.txt',delimiter=',')
#print(data)
print(type(data))
print(data.shape)
X = data[:, 0]#population
Y = data[:, 1]#profit
m = Y.size

plt.plot(X, Y, 'g+', label='point',linewidth=2)

plt.xlabel('population')
plt.ylabel('profit')
plt.title('')
#plt.ion()#Turn interactive mode on.
plt.legend()#显示图标


# ===================== Part 2: Gradient descent =====================
print('Running Gradient Descent...')

X=np.c_[np.ones(m),X] #y=theta1*x1+heta2*x2
alpha=0.01 #learning rate
iterations = 1500
theta=np.zeros(2) #theta1，theta2初始化为0
J_history=np.zeros(iterations)

print(theta.shape,J_history.shape)
theta , J_history=GD.gradientdescent(X,Y,theta,alpha,iterations)
print(theta,J_history,theta.shape)
X1=np.array(np.arange(0,20,0.01))

Y1=theta[0]+X1*theta[1]
plt.plot(X1, Y1, 'r-', label='line',linewidth=1)
plt.show()

input('Program paused. Press ENTER to continue')

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
x_theta0=np.linspace(-10,10,100)
y_theta0=np.linspace(-1,4,100)

# 对x、y数据执行网格化
xx,yy=np.meshgrid(x_theta0,y_theta0)
z_J=np.zeros(xx.shape)
for i in range(0,x_theta0.size):
	for j in range(0,y_theta0.size):
		theta_temp=np.array([x_theta0[i],y_theta0[j]])
		z_J[i][j]=GD.compute_cost(X,Y,theta_temp)

z_J = np.transpose(z_J)

fig = plt.figure(1)
ax = fig.gca(projection='3d')  #get a Axes3D object
ax.plot_surface(xx, yy, z_J)

plt.figure(2)

lvls = np.logspace(-2, 3, 30) #等高线的密度
plt.contour(xx, yy, z_J, levels=lvls, norm=LogNorm())#plot contour
plt.plot(theta[0], theta[1], c='r', marker="x")
print(theta)
#plt.plot(theta[0], theta[1], 'rx', ms=10, lw=2)
plt.show()