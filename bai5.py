import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
# Mo ta bo du lieu


data = [
[2.37319011, 1.71875981,0],
[1.51261889, 1.40558943,0],
[2.4696794 , 2.02144973,0],
[1.78736889, 1.29380961,0],
[1.81231157, 1.56119497,0],
[2.03717355, 1.93397133,0],
[1.53790057, 1.87434722,0],
[2.29312867, 2.76537389,0],
[1.38805594, 1.86419379,0],
[1.57279694, 0.90707347,0],
[3.42746579, 0.71254431,1],
[4.24760864, 2.39846497,1],
[3.33595491, 1.61731637,1],
[3.69420104, 1.94273986,1],
[4.53897645, 2.54957308,1],
[3.3071994 , 0.19362396,1],
[4.13924705, 2.09561534,1],
[4.47383468, 2.41269466,1],
[4.00512009, 1.89290099,1],
[4.28205624, 1.79675607,1]
]



# bai toan tuyen dung, Tu du lieu cua sinh vien bao gom
# toiec: diem toeic cua sinh vien
# gpa: diem chuan dau ra cua sinh vien
# work_experience: thoi gian lam viec cua sinh vien

# tu do thu nhan duoc ket qua tuyen dung trong admitted
# admitted = 1: co viec lam
# admitted = 0: that nghiep

# day la du lieu dung tu dien
# du lieu x gom 3 truong, du lieu y


X = [[x[0],x[1]] for x in data]
y = [x[2] for x in data]

# chia du lieu thanh 2 phan: train va test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)  #train is based on 75% of the dataset, test is based on 25% of dataset

# Tao va huan luyen mo hinh voi phan train
clf1 =  LinearSVC(random_state=0,C=0.1)
clf1.fit(X_train,y_train)


# Tao va huan luyen mo hinh voi phan train
clf2 =  LinearSVC(random_state=0,C=0.15)
clf2.fit(X_train,y_train)


# Tao va huan luyen mo hinh voi phan train
clf3 =  LinearSVC(random_state=0,C=0.9)
clf3.fit(X_train,y_train)





data1 = [
[2.37319011, 1.71875981,0],
[1.51261889, 1.40558943,0],
[2.4696794 , 2.02144973,0],
[1.78736889, 1.29380961,0],
[1.81231157, 1.56119497,0],
[2.03717355, 1.93397133,0],
[1.53790057, 1.87434722,0],
[2.29312867, 2.76537389,0],
[1.38805594, 1.86419379,0],
[1.57279694, 0.90707347,0]
]


data2 = [
[3.42746579, 0.71254431,1],
[4.24760864, 2.39846497,1],
[3.33595491, 1.61731637,1],
[3.69420104, 1.94273986,1],
[4.53897645, 2.54957308,1],
[3.3071994 , 0.19362396,1],
[4.13924705, 2.09561534,1],
[4.47383468, 2.41269466,1],
[4.00512009, 1.89290099,1],
[4.28205624, 1.79675607,1]
]



dx1 = [x[0] for x in data1]
dy1 = [y[1] for y in data1]

dx2 = [x[0] for x in data2]
dy2 = [y[1] for y in data2]


plt.scatter(dx1,dy1)
plt.scatter(dx2,dy2, color='red')


ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))


Z1 = clf1.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z1 = Z1.reshape(xx.shape)
plt.contour(xx, yy, Z1, colors='green', levels=[-1, 0, 1], alpha=0.5,
    linestyles=['--', '-', '--'])

Z2 = clf2.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z2 = Z2.reshape(xx.shape)
plt.contour(xx, yy, Z2, colors='lime', levels=[-1, 0, 1], alpha=0.5,
    linestyles=['--', '-', '--'])

Z3 = clf3.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z3 = Z3.reshape(xx.shape)
plt.contour(xx, yy, Z3, colors='aqua', levels=[-1, 0, 1], alpha=0.5,
    linestyles=['--', '-', '--'])


# plt.plot(X_train, logistic_regression.predict_proba(X_train), color='red')
# sns.regplot(x='price', y='buy',data = df,logistic=True, color='red')
plt.show()
