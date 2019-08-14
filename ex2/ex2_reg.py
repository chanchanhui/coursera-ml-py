import numpy as np 
from function_reg import (loadtxt,plotData,sigmoid,costFuncReg,
gradFuncReg,mapFeature,plotDecisonBoundary,prediction)
import	scipy.optimize as op
#plotData()
data_np=np.loadtxt("ex2data2.txt", delimiter=',')

m=np.size(data_np,0)
X=data_np[:,0:2]
y=data_np[:,2]
#extending the feture
x_result = mapFeature(data_np[:,0],data_np[:,1])

#initialize theta
theta=np.zeros((np.size(x_result,1),))
lamd = 1


error=costFuncReg(theta,x_result,y,lamd)
grad=gradFuncReg(theta,x_result,y,lamd)
#optimization
result = op.minimize(costFuncReg, x0=theta, method='BFGS', jac=gradFuncReg, args=(x_result, y, lamd))
theta=result.x

plotDecisonBoundary(theta,x_result,y)
val=prediction(data_np,theta)
print("precision:",(np.sum(val==y))/val.size)