
import numpy as np

from sklearn import datasets, linear_model
import numpy as np 
import matplotlib.pyplot as plt

# X: height (cm), 
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# y: weight (kg)
y = np.array([ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])



# fit the model by Linear Regression
regr = linear_model.LinearRegression()
regr.fit(X, y) 


# solution
print(" w_1 = ", regr.coef_[0], "w_0 = ", regr.intercept_)
w_0 = regr.intercept_
w_1 = regr.coef_[0]

x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1*x0

# Drawing the fitting line 
plt.plot(X, y.T, 'ro')     # data 
plt.plot(x0, y0)  
plt.plot( regr.intercept_,regr.coef_[0])               # the fitting line
plt.axis([140, 190, 45, 75])
plt.xlabel('Chiều cao (cm)')
plt.ylabel('Cân nặng  (kg)')
plt.show()
