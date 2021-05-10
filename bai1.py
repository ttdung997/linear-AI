
import numpy as np

from sklearn import datasets, linear_model
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures


# X: height (cm), 
X = np.array([[7.79667589,2.79825217,2.06174503,4.4713877 ,7.20443649,7.36014312,4.70688117,
	-0.40338389,4.72266607,1.20453709,6.07593449,7.69651292,3.89733971,4.7856351 ,
	-0.59932188,4.1507473 ,0.04186784,4.89562846,2.38650347,6.42758034]]).T
# y: weight (kg)
y = np.array([318.28185696,20.48143891,11.97873995
	,7.56902114,224.15497306,235.04403786,17.75040067
	,-107.86335911,1.1140603 ,-7.67492972,87.4263873 ,293.22569099,
	-11.49557421,6.4415876 ,-152.88870565,-4.95755333,-79.53431819,34.97246059,
	-4.50098315,95.09276699])



# fit the model by Linear Regression
regr = linear_model.LinearRegression()


poly_reg =  PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
print(X_poly)

regr.fit(X_poly, y) 


# solution
print(" w_1 = ", regr.coef_[0], "w_0 = ", regr.intercept_)
w_0 = regr.intercept_
w_1 = regr.coef_[0]

x0 = np.linspace(0, 10, 2)
y0 = w_0 + w_1*x0

# Drawing the fitting line 
plt.plot(X, y.T, 'ro')     # data 

# plt.plot(X , regr.predict( poly_reg.fit_transform(X)), color = 'blue')
plt.plot( regr.intercept_,regr.coef_[0])               # the fitting line
plt.axis([-10, 10, -300, 300])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
