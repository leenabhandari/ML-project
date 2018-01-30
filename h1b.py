from pandas import read_csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer

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
logistic.score(X_test,Y_test)  #0.8.0.84
full_time=ds[:,5]
full_time=data1[:, 5]
for i in range(full_time.size):
 if full_time[i]=='Y':
  full_time[i]=1
 else:
  full_time[i]=0
  
  
ds=data.values
X_data1=ds[:,6]
X_data2=np.c_[X_data1, full_time]
Y_data=ds[:,1]
X_train, X_test, Y_train, Y_test = train_test_split(X_data2, Y_data, test_size=0.33,random_state=7)


model= RandomForestClassifier(min_samples_leaf=50) #0.74
X_train, X_test, Y_train, Y_test = train_test_split(X_data2, Y_data, test_size=0.25,random_state=0)
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

lon=ds[:,9]
X_data3=np.c_[X_data2, lon]
lat=ds[:,10]
X_data4=np.c_[X_data3, lat]
imp=Imputer()
X_trans=imp.fit_transform(X_data4)
X_train, X_test, Y_train, Y_test = train_test_split(X_trans, Y_data, test_size=0.25,random_state=0)
model.fit(X_train,Y_train)
model.score(X_test,Y_test) #0.77, 0.8

model=tree.DecisionTreeClassifier(min_samples_leaf=50) #0.84
X_soc= ds[:,3]
for i in range(ds.size):
  if X_soc[i]=='ADVERTISING AND PROMOTIONS MANAGERS':
   X_soc[i]=1
  elif X_soc[i]=='BIOCHEMISTS AND BIOPHYSICISTS':
   X_soc[i]=2
  elif X_soc[i]=='CHIEF EXECUTIVES':
   X_soc[i]=3
  elif X_soc[i]=='FINANCIAL MANAGERS':
   X_soc[i]=4
  elif X_soc[i]=='GENERAL AND OPERATIONS MANAGER':
   X_soc[i]=5
  elif X_soc[i]=='GENERAL AND OPERATIONS MANAGERS':
   X_soc[i]=5
  elif X_soc[i]=='GENERAL AND OPERATIONS MANAGERSE':
   X_soc[i]=5
  elif X_soc[i]=='MARKETING MANAGERS':
   X_soc[i]=6
  elif X_soc[i]=='PUBLIC RELATIONS SPECIALISTS':
   X_soc[i]=7

corpus=['ADVERTISING AND PROMOTIONS MANAGERS','BIOCHEMISTS AND BIOPHYSICISTS','CHIEF EXECUTIVES','FINANCIAL MANAGERS','GENERAL AND OPERATIONS MANAGER','MARKETING MANAGERS','PUBLIC RELATIONS SPECIALISTS']
vectorizer = CountVectorizer()
vectorizer.fit_transform(corpus)
ds=data.values

for i in range(2999):
  if ds[i,3]=='GENERAL AND OPERATIONS MANAGERS':
   ds[i,3]='GENERAL AND OPERATIONS MANAGER'
  if ds[i,3]=='GENERAL AND OPERATIONS MANAGERSE':
   ds[i,3]='GENERAL AND OPERATIONS MANAGER'
   
X_soc=vectorizer.transform(ds[:,3]).toarray()
X_trans1=np.c_[X_soc, X_trans]
X_train, X_test, Y_train, Y_test = train_test_split(X_trans1, Y_data, test_size=0.25,random_state=0) #0.84

