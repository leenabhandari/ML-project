from pandas import read_csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt
import string


fname='h1b_sample3.csv'
data=read_csv(fname)
data=read_csv('h1b_kaggle1.csv')

################
data1=np.array(data)
X_data=data1[:, 6]
Y_data=data1[:, 1]
X_data=X_data.reshape(-1,1)
Y_data=Y_data.reshape(-1,1)
X_train=X_data[: -70]
X_test=X_data[-70 :]
Y_train=Y_data[: -70]
Y_test=Y_data[-70 :]
model=KNeighborsClassifier()
model.fit(X_train,Y_train)
res=model.predict(X_test).reshape(-1,1)
match=0
for i in range(Y_test.size):
 if res[i]==Y_test[i]:
   match=match+1
   
err=Y_test.size-match

model=SVC()
model.score(X_test,Y_test) #0.6714285714285714

model = linear_model.LogisticRegression(C=1e5)
model.fit(X_train, Y_train)
model.score(X_test,Y_test)  #0.8.0.84

#########
ds=data.values

full_time=ds[:,5]

for i in range(full_time.size):
 if full_time[i]=='Y':
  full_time[i]=1
 else:
  full_time[i]=0
 

X_data1=ds[:,6]
X_data2=np.c_[X_data1, full_time]
Y_data=ds[:,1]
X_train, X_test, Y_train, Y_test = train_test_split(X_data2, Y_data, test_size=0.10,random_state=7)


model= RandomForestClassifier() #0.74
X_train, X_test, Y_train, Y_test = train_test_split(X_data2, Y_data, test_size=0.25,random_state=0)
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

lon=ds[:,9]
X_data3=np.c_[X_data2, lon]
lat=ds[:,10]
X_data4=np.c_[X_data3, lat]
imp=Imputer()
X_trans=imp.fit_transform(X_data4)
X_train, X_test, Y_train, Y_test = train_test_split(X_trans, Y_data, test_size=0.10,random_state=7)
model.fit(X_train,Y_train)
model.score(X_test,Y_test) #0.77, 0.8

model=tree.DecisionTreeClassifier(min_samples_leaf=50)
model=tree.DecisionTreeClassifier() #0.84
X_soc= ds[:,3]

corp=np.unique(ds[:,3])
corpus=['ADVERTISING AND PROMOTIONS MANAGERS','BIOCHEMISTS AND BIOPHYSICISTS','CHIEF EXECUTIVES','FINANCIAL MANAGERS','GENERAL AND OPERATIONS MANAGER','MARKETING MANAGERS','PUBLIC RELATIONS SPECIALISTS']

ds=data.values



soc=ds[:,3]   
for i in range(soc.size):
  soc[i]=string.replace(soc[i],'MANAGERS','MANAGER')
  soc[i]=string.replace(soc[i],'&','AND')
  soc[i]=string.replace(soc[i],'FUNDRAISING','FUND RAISING')
  soc[i]=string.replace(soc[i],'INFORMATON','INFORMATION')
  soc[i]=string.replace(soc[i],'MANGERS','MANAGER')
  soc[i]=string.replace(soc[i],'MANAGERE','MANAGER')


sfit=np.unique(soc)

vectorizer = CountVectorizer()
vectorizer.fit_transform(sfit)  
X_soc=vectorizer.transform(soc).toarray()
X_trans1=np.c_[X_soc, X_trans]
X_train, X_test, Y_train, Y_test = train_test_split(X_trans1, Y_data, test_size=0.25,random_state=0) #0.84

#######
count=0
for i in range(Y_test.size):
  if model.predict(X_test)[i]=='CERTIFIED':
   count=count+1
   
for i in range(Y_test.size):
  if model.predict(X_test)[i]==Y_test[i] and Y_test[i]!='CERTIFIED':
   count=count+1
##########
ws=ds[:,8] 
c1=np.array([])
for i in range(ws.size):
  a,b=ws[i].split(",")
  c1=np.append(c1,b)

workstate=np.unique(d)
vec = CountVectorizer()
vec.fit_transform(workstate)
X_ws=vec.transform(d).toarray()
X_new2=np.c_[X_new, X_ws]
X_train, X_test, Y_train, Y_test = train_test_split(X_new2, Y_data, test_size=0.10,random_state=7)
model=tree.DecisionTreeClassifier(min_samples_leaf=10) #0.82

ws=ws.astype(str)
c=np.char.split(ws,',')
d=[i[1] for i in c]
workstate=set(d)
hv = HashingVectorizer(n_features=10)
X_ws=hv.transform(d).toarray()
X_new=np.c_[X_data1,X_ws]
imp=Imputer()
X_trans=imp.fit_transform(X_new)
##########
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y_data, cv=kfold)
predicted=cross_val_predict(model,X_trans,Y_data,cv=kfold)
metrics.accuracy_score(Y_data, predicted)
results.mean()