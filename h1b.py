from pandas import read_csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
fname='h1b_kaggle.csv'
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