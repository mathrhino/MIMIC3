import pymysql
from pandas import DataFrame as df
import pandas as pd
import numpy as np
from  datetime import datetime
import time
from sklearn.preprocessing import OneHotEncoder

connection = pymysql.connect(host='192.168.56.102', user='dba', password='mysql', db='mimiciiiv14', charset='utf8')
cursor = connection.cursor()
sql = "select DOB, DOD from PATIENTS "
cursor.execute(sql)
data = cursor.fetchall()

data = df(data)

days = []
for i in range(0, len(data)):
    start_time = data.iloc[:,0]
    element = start_time[i].to_pydatetime()
    timestamp = datetime.timestamp(element)
days = df(days)

print(days)



'''
## float --> int ##
data = data.astype('int')

def to_int(data)
    data = data.astype('int')
    
    return data

## Timestamp days -- Age ##
days = []
for i in range(0, len(data)):
    start_time = data.iloc[:,0]
    start_time = start_time[i].to_pydatetime()
    finish_time= data.iloc[:,1]
    finish_time = finish_time[i].to_pydatetime()
    result = ((finish_time- start_time).days)/365
    days.append(result)
days = df(days)

def cal_age(start_time, finish_time):
    start_time = start_time.to_pydatetime()
    finish_time = finish_time.to_pydatetime()
    result = ((finish_time- start_time).days) / 365
    
    return result

## Timestamp days -- Age ##
days = []
for i in range(0, len(data)):
    start_time = data.iloc[:,0]
    start_time = start_time[i].to_pydatetime()
    finish_time= data.iloc[:,1]
    finish_time = finish_time[i].to_pydatetime()
    result = (finish_time- start_time).days
    days.append(result)
days = df(days)

def cal_days(start_time, finish_time):
    start_time = start_time.to_pydatetime()
    finish_time = finish_time.to_pydatetime()
    result = (finish_time- start_time).days
    
    return result
    

## My own categorize
num_unique = data.nunique().astype('int')
unique = pd.unique(data.iloc[:,0]).tolist()
for i in range(0, num_unique[0]):
    print(i, unique[i])
    data = data.replace(to_replace=unique[i], value=i, method='pad')
    
def categolize(data):
    num_unique = data.nunique().astype('int')
    unique = pd.unique(data.iloc[:,0]).tolist()
    for i in range(0, num_unique[0]):
        print(i, unique[i])
        data = data.replace(to_replace=unique[i], value=i, method='pad')
    
    return data

## One-hot-encoding
enc= OneHotEncoder()
enc.fit(data)
data = enc.transform(data)

def oneHotEncoding(data):
    enc= OneHotEncoder()
    enc.fit(data)
    data = enc.transform(data)
    
    return data

'''