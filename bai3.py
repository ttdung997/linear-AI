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
[ 1.92306918, 0.775673 ,0],
[ 2.90509186, 1.7997662 ,0],
[ 1.58909188, 1.16143907,0],
[ 2.76874122, 1.07663514,0],
[ 2.52416203, 1.53020387,0],
[ 2.02414192, 1.80692632,0],
[ 2.49174878, 2.62943405,0],
[ 1.11439322, 2.88348991,0],
[ 2.62561276, 2.89077234,0],
[ 3.27183166, 0.75454543,0],
[ 3.55617919, 0.66250438,0],
[ 1.45945603, 2.28222634,0],
[ 2.87575608, 2.52637908,0],
[ 2.30375703, 2.46497356,0],
[ 1.08925412, 2.01982447,0],
[ 4.09096119, -0.08330889,0],
[ 1.80350003, 1.91837255,0],
[ 1.25827634, 1.8856175 ,0],
[ 2.08532169, 1.79005729,0],
[ 1.9340609 , 1.09208652,0],
[4.49775285, 1.46545116,1],
[3.07311718, 3.76212796,1],
[3.52528933, 1.59844519,1],
[2.39091046, 2.33431976,1],
[3.12302646, 1.41945943,1],
[4.62359547, 2.44921113,1],
[3.38696098, 2.46494505,1],
[4.36167918, 1.88637824,1],
[3.21261415, 2.40558547,1],
[2.71754956, 2.33882965,1],
[4.01073111, 0.96947283,1],
[3.11892586, 4.10522222,1],
[2.29402636, 2.07905375,1],
[3.7365999 , 2.41022672,1],
[4.29699439, 2.79542218,1],
[4.63617269, 2.00962462,1],
[2.84870815, 1.77280105,1],
[4.45368062, 2.44805003,1],
[4.97052399, 1.86194687,1],
[3.01324102, 1.54377016,1]
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
clf =  LinearSVC(random_state=0)
clf.fit(X_train,y_train)

# Kiem thu ket qua voi tap test
y_pred=clf.predict(X_test)

# load metrics de danh gia do chinh xac cua mo hinh 

report = metrics.classification_report(y_test,y_pred,digits=4) 


print (report)

data1 = [
[ 1.92306918, 0.775673 ,0],
[ 2.90509186, 1.7997662 ,0],
[ 1.58909188, 1.16143907,0],
[ 2.76874122, 1.07663514,0],
[ 2.52416203, 1.53020387,0],
[ 2.02414192, 1.80692632,0],
[ 2.49174878, 2.62943405,0],
[ 1.11439322, 2.88348991,0],
[ 2.62561276, 2.89077234,0],
[ 3.27183166, 0.75454543,0],
[ 3.55617919, 0.66250438,0],
[ 1.45945603, 2.28222634,0],
[ 2.87575608, 2.52637908,0],
[ 2.30375703, 2.46497356,0],
[ 1.08925412, 2.01982447,0],
[ 4.09096119, -0.08330889,0],
[ 1.80350003, 1.91837255,0],
[ 1.25827634, 1.8856175 ,0],
[ 2.08532169, 1.79005729,0],
[ 1.9340609 , 1.09208652,0],
]

data2 = [[4.49775285, 1.46545116,1],
[3.07311718, 3.76212796,1],
[3.52528933, 1.59844519,1],
[2.39091046, 2.33431976,1],
[3.12302646, 1.41945943,1],
[4.62359547, 2.44921113,1],
[3.38696098, 2.46494505,1],
[4.36167918, 1.88637824,1],
[3.21261415, 2.40558547,1],
[2.71754956, 2.33882965,1],
[4.01073111, 0.96947283,1],
[3.11892586, 4.10522222,1],
[2.29402636, 2.07905375,1],
[3.7365999 , 2.41022672,1],
[4.29699439, 2.79542218,1],
[4.63617269, 2.00962462,1],
[2.84870815, 1.77280105,1],
[4.45368062, 2.44805003,1],
[4.97052399, 1.86194687,1],
[3.01324102, 1.54377016,1]
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


Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
    linestyles=['--', '-', '--'])
# plt.plot(X_train, logistic_regression.predict_proba(X_train), color='red')
# sns.regplot(x='price', y='buy',data = df,logistic=True, color='red')
plt.show()
