from pandas import read_csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
#from sklearn import linear_model
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
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes  import GaussianNB
#from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
#import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
#from pickle import dump
#from pickle import load
from sklearn.model_selection import GridSearchCV
#import string
import matplotlib.pyplot as plt
from matplotlib import pyplot
from dataprep import data2
#from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import pandas as pd





data2=read_csv('data2.csv')
data2=np.array(data2)

#separate out X&Y
Y_new=data2[:,1]
X_new=data2[:,2:7]  

#dimension reduction
pca=PCA(n_components=2)
pca.fit(X_new)
X2=pca.transform(X_new)
#kfold = KFold(n_splits=10, random_state=7)  
X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y_new, test_size=0.10,random_state=7)
# X_wage=ds[:,6] #prevailing wage



model=RandomForestClassifier()
prams={'max_depth':[None,1,2,3,4],'n_estimators':[10,20,30,50,1,100]}
grid=GridSearchCV(model,prams)
grid.fit(X_new,Y_new)
print model
print grid.best_score_ #0.33001699830017
print grid.best_estimator_.n_estimators #100

model=DecisionTreeClassifier()
prams={'min_samples_leaf':(1,2,3,5),'max_depth':[None,1,2,3]}
grid=GridSearchCV(model,prams)
grid.fit(X_new,Y_new)
print model
print grid.best_score_ #0.3263173682631737
print grid.best_estimator_.min_samples_leaf #1

model=AdaBoostClassifier()
prams={'n_estimators':[50,10,100],'learning_rate':[1,2]}
grid=GridSearchCV(model,prams)
grid.fit(X_new,Y_new)
print model
print grid.best_score_  #0.31916808319168083
print grid.best_estimator_.learning_rate #1
print grid.best_estimator_.n_estimators #100

model=KNeighborsClassifier()
prams={'n_neighbors':[4,5,8]}
grid=GridSearchCV(model,prams)
grid.fit(X_new,Y_new)
print model
print grid.best_score_ #0.3191930806919308
print grid.best_estimator_.n_neighbors #4

#plt.plot(np.corrcoef(X_new.astype(float)[0:25,:]))

#plot correlation score
names=['wage','lon','lat','full','states']
datafi=pd.read_csv("data2.csv",names=names)
correlations=datafi.corr()
plt.plot(correlations)

#plot correlations
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

