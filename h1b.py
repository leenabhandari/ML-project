from pandas import read_csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
fname='h1b_sample.csv'
data=read_csv(fname)
data1=np.array(data)
X_data=data1[:, 6]
Y_data=data1[:, 1]
X_data=X_data.reshape(-1,1)
Y_data=Y_data.reshape(-1,1)
X_train=X_data[: -70]
X_test=X_data[-70 :]
Y_train=Y_data[: -70]
Y_test=Y_data[-70 :]
knn=KNeighborsClassifier()
knn.fit(X_train,Y_train)
res=knn.predict(X_test).reshape(-1,1)
match=0
for i in range(Y_test.size):
 if res[i]==Y_test[i]:
   match=match+1
   
err=Y_test.size-match

knn.score(X_test,Y_test) #0.6714285714285714

logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(X_train, Y_train)
logistic.score(X_test,Y_test)  #0.8

full_time=data1[:, 5]
for i in range(full_time.size):
 if full_time[i]=='Y':
  full_time[i]=1
 else:
  full_time[i]=0
  
  
ds=data.values
X_data1=ds[:,6]
X_data2=np.c_[X_data1, full_time]

X_train, X_test, Y_train, Y_test = train_test_split(X_data2, Y_data, test_size=0.33,random_state=7)