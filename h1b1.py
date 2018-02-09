from pandas import read_csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
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
from pickle import dump
from pickle import load
import string

data=read_csv('h1b_kaggle1.csv')
index_miss=data.lat.isnull()
data2=data[index_miss!= True]
data2.to_csv("h1b_data.csv")
ds=data2.values
Y_data=ds[:,1]

X_data1=ds[:,6]

lat=ds[:,10]
X_data2=np.c_[X_data1, lat]

ws=ds[:,8] 
ws=ws.astype(str)
c=np.char.split(ws,',')
d=[i[1] for i in c]
workstate=set(d)
vec = CountVectorizer()
vec.fit_transform(workstate)
X_ws=vec.transform(d).toarray()
X_data3=np.c_[X_data1,X_ws]

lon=ds[:,9]
X_data4=np.c_[X_data1, lon]


kfold = KFold(n_splits=10, random_state=7)
models=[KNeighborsClassifier(),SGDClassifier(),LogisticRegression(),AdaBoostClassifier(),DecisionTreeClassifier(),RandomForestClassifier(), MultinomialNB()]

#for model in models:
#	predicted=cross_val_predict(model,X_data3,Y_data,cv=kfold)
#	print model
#	print metrics.accuracy_score(Y_data, predicted)
#	print ""

X_train, X_test, Y_train, Y_test = train_test_split(X_data2, Y_data, test_size=0.10,random_state=7)

for model in models:
	filename=str(model)[0:15]+ ".sav"
	model.fit(X_train,Y_train)
	dump(model, open(filename, 'wb'))
	loaded_model = load(open(filename, 'rb'))
	result = loaded_model.score(X_test, Y_test)
	print model
	print(result)

	
