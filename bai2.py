import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Mo ta bo du lieu

# bai toan tuyen dung, Tu du lieu cua sinh vien bao gom
# toiec: diem toeic cua sinh vien
# gpa: diem chuan dau ra cua sinh vien
# work_experience: thoi gian lam viec cua sinh vien

# tu do thu nhan duoc ket qua tuyen dung trong admitted
# admitted = 1: co viec lam
# admitted = 0: that nghiep

# day la du lieu dung tu dien
candidates = {'price': [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50,
4.00, 4.25, 4.50, 4.75, 5.00, 5.50],
              'buy': [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
               }

# chuyen du lieu tu dien ve mang data
df = pd.DataFrame(candidates,columns= ["price","buy"])
df.to_csv("data.csv",index=False)

# du lieu x gom 3 truong, du lieu y
X = df[['price']]
y = df['buy']  

# chia du lieu thanh 2 phan: train va test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)  #train is based on 75% of the dataset, test is based on 25% of dataset

# Tao va huan luyen mo hinh voi phan train
logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)

# Kiem thu ket qua voi tap test
y_pred=logistic_regression.predict(X_test)

# load metrics de danh gia do chinh xac cua mo hinh 

report = metrics.classification_report(y_test,y_pred,digits=4) 
# classification_report cho phep danh gia ket qua cho bai toan phan lop
# Cac ban tim doc cac chi so precision, recall (tim thong qua google)
# de hien ro hon ve bai toan phan lop
print (report)

#tuy chinh nguong threshold

y_pred =[]
y_statitic=logistic_regression.predict_proba(X_test)
threshold = 0.2
for pre in y_statitic:
	if pre[0] > threshold:
		y_pred.append(0)
	else:
		y_pred.append(1)

# danh gia lai voi nguong moi
report = metrics.classification_report(y_test,y_pred,digits=4) 

print (report)



plt.scatter(X_train, y_train)
# plt.plot(X_train, logistic_regression.predict_proba(X_train), color='red')
sns.regplot(x='price', y='buy',data = df,logistic=True, color='red')
plt.show()