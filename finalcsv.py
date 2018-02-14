from pandas import read_csv
import numpy as np



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
data.drop(data.columns[[0, 1]], axis=1)
data.to_csv("newdata.csv")


