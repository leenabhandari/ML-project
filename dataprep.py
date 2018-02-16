import pandas as pd
from pandas import read_csv
import numpy as np

#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
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
data=data.drop(data.columns[[0]], axis=1)

data1=data
#data1=read_csv('newdata.csv')
data1=data1.reindex(np.random.permutation(data1.index)) #shuffle all data
ds=data1.values
Y_data=ds[:,0] #CASE_STATUS

full_time=ds[:,4]
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
data_soc=np.array([])


#append rows ( 10000 of each category)
count_cert=0
for i in range(Y_data.size):
 if (count_cert>10000):
  break
 elif (Y_data[i]=='CERTIFIED'):
  data_y=np.append(data_y,Y_data[i])
  data_wage=np.append(data_wage,ds[i,5])
  data_lon=np.append(data_lon,ds[i,8])
  data_lat=np.append(data_lat,ds[i,9])
  data_full=np.append(data_full,full_time[i])
  data_ws=np.append(data_ws,ds[i,7])
  data_soc=np.append(data_soc,ds[i,2])
  count_cert=count_cert+1

count_with=0
for i in range(Y_data.size):
 if (count_with>10000):
  break
 elif (Y_data[i]=='WITHDRAWN'):
  data_y=np.append(data_y,Y_data[i])
  data_wage=np.append(data_wage,ds[i,5])
  data_lon=np.append(data_lon,ds[i,8])
  data_lat=np.append(data_lat,ds[i,9])
  data_full=np.append(data_full,full_time[i])
  data_ws=np.append(data_ws,ds[i,7])
  data_soc=np.append(data_soc,ds[i,2])
  count_with=count_with+1		


count_certwith=0
for i in range(Y_data.size):
 if (count_certwith>10000):
  break
 elif (Y_data[i]=='CERTIFIED-WITHDRAWN'):
  data_y=np.append(data_y,Y_data[i])
  data_wage=np.append(data_wage,ds[i,5])
  data_lon=np.append(data_lon,ds[i,8])
  data_lat=np.append(data_lat,ds[i,9])
  data_full=np.append(data_full,full_time[i])
  data_ws=np.append(data_ws,ds[i,7])
  data_soc=np.append(data_soc,ds[i,2])
  count_certwith=count_certwith+1

  
count_den=0
for i in range(Y_data.size):
 if (count_den>10000):
  break
 elif (Y_data[i]=='DENIED'):
  data_y=np.append(data_y,Y_data[i])
  data_wage=np.append(data_wage,ds[i,5])
  data_lon=np.append(data_lon,ds[i,8])
  data_lat=np.append(data_lat,ds[i,9])
  data_full=np.append(data_full,full_time[i])
  data_ws=np.append(data_ws,ds[i,7])
  data_soc=np.append(data_soc,ds[i,2])
  count_den=count_den+1
  

ws=data_ws.astype(str) #convert data_ws to string and store in ws
#worksites=np.unique(data_ws) #pull out unique worksites
ws_split=np.char.split(ws,',') #split into city and state
data_states=[i[1] for i in ws_split] #store states
#workstate=set(data_states) #consider unique states
#vec = CountVectorizer()
#vec.fit_transform(workstate) #fit unique states
#X_ws=vec.transform(data_states).toarray() #transform states and convert to array

hv=HashingVectorizer(n_features=1)
X_ws=hv.transform(data_states).toarray()
  
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

df = pd.DataFrame(data2)
df.to_csv("data2.csv")