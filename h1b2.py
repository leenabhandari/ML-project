from pandas import read_csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
#from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.svm import SVC
#from sklearn.preprocessing import Imputer
#from sklearn.feature_extraction import FeatureHasher
#from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes  import GaussianNB
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
#from pickle import dump
#from pickle import load
#from sklearn.model_selection import GridSearchCV
import string


data=read_csv('h1b_kaggle.csv')

#remove data with missiing values
index_miss=data.lat.isnull()
data=data[index_miss!= True]
index_miss=data.lon.isnull()
data=data[index_miss!= True]
index_miss=data.PREVAILING_WAGE.isnull()
data=data[index_miss!= True]
index_miss=data.WORKSITE.isnull()
data=data[index_miss!= True]
index_miss=data.SOC_NAME.isnull()
data=data[index_miss!= True]
data.to_csv("h1b_data.csv")
#ds=data.values
#np.random.shuffle(ds)

data1=data
data1=data1.reindex(np.random.permutation(data1.index)) #shuffle all data

ds=data1.values
Y_data=ds[:,1] #CASE_STATUS

full_time=ds[:,5]
#convert full_time value to 0 & 1
for i in range(full_time.size):
 if full_time[i]=='Y':
  full_time[i]=1
 else:
  full_time[i]=0

#initialize arrays
data2=np.array([])
data_y=np.array([]) #CASE_STATUS
data_wage=np.array([]) #PREVAILING_WAGE
data_lon=np.array([]) #lon
data_lat=np.array([]) #lat
data_full=np.array([]) #FULL_TIME
data_ws=np.array([]) #WORKSITE
#data_soc=np.array([])


#append rows ( 10000 of each category)
count_cert=0
for i in range(Y_data.size):
 if (count_cert>10000):
  break
 elif (Y_data[i]=='CERTIFIED'):
  data_y=np.append(data_y,Y_data[i])
  data_wage=np.append(data_wage,ds[i,6])
  data_lon=np.append(data_lon,ds[i,9])
  data_lat=np.append(data_lat,ds[i,10])
  data_full=np.append(data_full,full_time[i])
  data_ws=np.append(data_ws,ds[i,8])
  #data_soc=np.append(data_soc,ds[i,3])
  count_cert=count_cert+1

count_with=0
for i in range(Y_data.size):
 if (count_with>10000):
  break
 elif (Y_data[i]=='WITHDRAWN'):
  data_y=np.append(data_y,Y_data[i])
  data_wage=np.append(data_wage,ds[i,6])
  data_lon=np.append(data_lon,ds[i,9])
  data_lat=np.append(data_lat,ds[i,10])
  data_full=np.append(data_full,full_time[i])
  data_ws=np.append(data_ws,ds[i,8])
  #data_soc=np.append(data_soc,ds[i,3])
  count_with=count_with+1		


count_certwith=0
for i in range(Y_data.size):
 if (count_certwith>10000):
  break
 elif (Y_data[i]=='CERTIFIED-WITHDRAWN'):
  data_y=np.append(data_y,Y_data[i])
  data_wage=np.append(data_wage,ds[i,6])
  data_lon=np.append(data_lon,ds[i,9])
  data_lat=np.append(data_lat,ds[i,10])
  data_full=np.append(data_full,full_time[i])
  data_ws=np.append(data_ws,ds[i,8])
  #data_soc=np.append(data_soc,ds[i,3])
  count_certwith=count_certwith+1

  
count_den=0
for i in range(Y_data.size):
 if (count_den>10000):
  break
 elif (Y_data[i]=='DENIED'):
  data_y=np.append(data_y,Y_data[i])
  data_wage=np.append(data_wage,ds[i,6])
  data_lon=np.append(data_lon,ds[i,9])
  data_lat=np.append(data_lat,ds[i,10])
  data_full=np.append(data_full,full_time[i])
  data_ws=np.append(data_ws,ds[i,8])
  #data_soc=np.append(data_soc,ds[i,3])
  count_den=count_den+1
  

ws=data_ws.astype(str) #convert data_ws to string and store in ws
#worksites=np.unique(data_ws) #pull out unique worksites
ws_split=np.char.split(ws,',') #split into city and state
data_states=[i[1] for i in ws_split] #store states
workstate=set(data_states) #consider unique states
vec = CountVectorizer()
vec.fit_transform(workstate) #fit unique states
X_ws=vec.transform(data_states).toarray() #transform states and convert to array

  
for i in range(data_soc.size):
  data_soc[i]=string.replace(data_soc[i],'MANAGERS','MANAGER')
  data_soc[i]=string.replace(data_soc[i],'&','AND')
  data_soc[i]=string.replace(data_soc[i],'FUNDRAISING','FUND RAISING')
  data_soc[i]=string.replace(data_soc[i],'INFORMATON','INFORMATION')
  data_soc[i]=string.replace(data_soc[i],'MANGERS','MANAGER')
  data_soc[i]=string.replace(data_soc[i],'MANAGERE','MANAGER')
  data_soc[i]=string.replace(data_soc[i],'SPECIALISTS','SPECIALIST')
  data_soc[i]=string.replace(data_soc[i],'DEVELOPERS','DEVELOPER')
  data_soc[i]=string.replace(data_soc[i],',','')
  data_soc[i]=string.replace(data_soc[i],'R & D','R&D')

#append all colums to data2
data2=np.c_[data_y, data_wage]
data2=np.c_[data2, data_lon]
data2=np.c_[data2, data_lat]
data2=np.c_[data2, data_full]
data2=np.c_[data2, X_ws]
np.random.shuffle(data2)

#separate out X&Y
Y_new=data2[:,0]
X_new=data2[:,1:63]  
#kfold = KFold(n_splits=10, random_state=7)  
X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y_new, test_size=0.10,random_state=7)
# X_wage=ds[:,6] #prevailing wage

model=RandomeForestClassifier()
prams={'max_depth':[None,1,2,3,4],'n_estimators':[10,20,30,50,1,100]}
grid=GridSearchCV(model,prams)
grid.fit(X_new,Y_new)
print grid.best_score_ #0.33001699830017
print grid.best_estimator_.n_estimators #100

model=DecisionTreeClassifier()
prams={'min_samples_leaf':(1,2,3,5),'max_depth':[None,1,2,3]}
grid.best_score_ #0.3263173682631737
grid.best_estimator_.min_samples_leaf #1

model=AdaBoostClassifier()
prams={'n_estimators':[50,10,100],'learning_rate':[1,2]}
grid.best_score_  #0.31916808319168083
grid.best_estimator_.learning_rate #1
grid.best_estimator_.n_estimators #100

model=KNeighborsClassifier()
prams={'n_neighbors':[4,5,8]}
grid.best_score_ #0.3191930806919308
grid.best_estimator_.n_neighbors #4