# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pymysql
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import sklearn
import sklearn.metrics
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
import datetime
import time
from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN
from sklearn.model_selection import train_test_split

def changeValue(datasetX):

    print(datasetX.columns)
    if 'LOS' in datasetX.columns:
        datasetX['LOS'] = datasetX.LOS.astype(int)
    if 'PATIENTWEIGHT' in datasetX.columns:
        datasetX['PATIENTWEIGHT'] = datasetX.PATIENTWEIGHT.astype(int)
    ####### 이 부분 어떻게 해야할 지??? ##########

    time_col = ['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'EDREGTIME', 'EDOUTTIME', 'CHARTDATE', 'INTIME', 'OUTTIME', 'DOB', 'DOD', 'DOD_HOSP', 'DOD_SSN']
    # time_start = {'DISCHTIME':'ADMITTIME', 'DEATHTIME':'ADMITTIME', 'EDOUTTIME':'EDREGTIME'
    #     , 'OUTTIME':'INTIME', 'DOD':'DOB', 'DOD_HOSP':'DOB', 'DOD_SSN':'DOB'}

    datasetX = df(datasetX)
    for col in datasetX.columns:
        if col in time_col:
            start_time = datasetX.loc[:, col]
            for i in range(0, len(datasetX)):
                time_S = start_time[i].to_pydatetime()
                datasetX.loc[i, col] = time_S.day
    return datasetX

    # start_time = datasetX.loc[:, 'DOB']
    # for i in range(0, len(datasetX)):
    #     time_S = start_time[i].to_pydatetime()
    #     datasetX.loc[i,'DOB'] = time_S.day

def cal_age(start_time, finish_time):
    start_time = start_time.to_pydatetime()
    finish_time = finish_time.to_pydatetime()
    result = ((finish_time - start_time).days) / 365
    return result


def oneHotEncoding(data):
    enc = OneHotEncoder()
    enc.fit(data[['ADMISSION_TYPE', 'GENDER']])
    datas = enc.transform(data[['ADMISSION_TYPE', 'GENDER']])
    pd.concat()
    return data


def categorize(datas):
    data = datas['ADMISSION_TYPE']
    num_unique = data.nunique().astype('int')
    unique = pd.unique(data.iloc[:, 0]).tolist()
    for i in range(0, num_unique[0]):
        print(i, unique[i])
        data = data.replace(to_replace=unique[i], value=i, method='pad')
    datas['ADMISSION_TYPE'] = data

    return datas


def cal_days(data):
    days = []
    for i in range(0, len(data)):
        start_time = data.loc[i, 'ADMITTIME']
        start_time = start_time.to_pydatetime()
        finish_time = data.loc[i, 'DISCHTIME']
        finish_time = finish_time.to_pydatetime()
        result = (finish_time - start_time).days
        days.append(result)
    days = df(days)
    return days


class MIMIC3(torch.utils.data.Dataset):
    def __init__(self, col_list="default", attr_list="default", categorize = None, data_opt = 'train', scaling = 'mean-std', holdout = 0.3, random_seed = 42, oversampler = None):
        self.col_list = col_list;
        self.attr_list = attr_list

        if(col_list == 'default'): self.col_list = ['ADMISSIONS', 'ICUSTAYS', 'INPUTEVENTS_MV', 'PATIENTS']
        if(attr_list == 'default'): self.attr_list = {'ADMISSIONS': ['ADMISSION_TYPE', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME'], 'ICUSTAYS': ['LOS'],
                                                      'INPUTEVENTS_MV': ['PATIENTWEIGHT'],'PATIENTS': ['DOB', 'GENDER']}

        for y in ['ADMITTIME', 'DISCHTIME', 'DEATHTIME'] :
            if y not in self.attr_list['ADMISSIONS']: self.attr_list['ADMISSIONS'].append(y)

        self.datasetX = None
        self.datasetY = None

        col_list = self.col_list
        conn = pymysql.connect(host='192.168.56.104', user='dba', password='mysql', db='mimiciiiv14', charset='utf8')
        curs = conn.cursor(pymysql.cursors.DictCursor)
        # Select LABEVENTS,SUBJECT_ID FROM LABEVENTS JOIN PATIENTS on LABEVENTS.SUBJECT_ID = PATIENTS.SUBJECT_ID JOIN ADMISSIONS on PATIENTS.SUBJECT_ID = ADMISSIONS.SUBJECT_ID
        sql_line = 'SELECT'
        for col in col_list:
            for attr in self.attr_list[col]:
                sql_line += ' ,' + col + '.' + attr
        sql_line += ' FROM ' + col_list[0]
        sql_line = sql_line[:7] + sql_line[8:]

        prev = col_list[0]
        for col in col_list[1:]:
            if col != 'PATIENTS':
                sql_line += ' JOIN {0} on {1}.SUBJECT_ID = {0}.SUBJECT_ID and {1}.HADM_ID = {0}.HADM_ID'.format(col,prev)
            else:
                sql_line += ' JOIN {0} on {1}.SUBJECT_ID = {0}.SUBJECT_ID'.format(col, prev)
            col_list[0] = col
        sql_line += ';'
        curs.execute(sql_line)
        result = curs.fetchall()
        print(df(result))
        # 여기부터
        self.datasetX = df(result)
        self.datasetY = self.datasetX[['ADMITTIME', 'DISCHTIME', 'DEATHTIME']]
        self.datasetX = self.datasetX.drop(['ADMITTIME', 'DISCHTIME', 'DEATHTIME'], axis=1)

        self.datasetX = changeValue(self.datasetX)#.to_numpy()
        for i in self.datasetX.columns:
            if self.datasetX[i].dtype == object:
                self.datasetX = pd.concat([self.datasetX, pd.get_dummies(self.datasetX[i], prefix = i)], axis = 1)
                del(self.datasetX[i])

        if((type(categorize) is list) and categorize != None):
            for i in categorize :
                self.datasetX = pd.concat([self.datasetX, pd.get_dummies(self.datasetX[i], prefix=i)], axis=1)
                del (self.datasetX[i])
        print(self.datasetX.shape)
        print(self.datasetX)
        self.datasetY = cal_days(self.datasetY)
        self.datasetY = self.datasetY.fillna(self.datasetY.mean())
        self.datasetY = self.datasetY.to_numpy()
        # 여기까지 뜯어 고쳐야함.

        X_train, X_test, y_train, y_test = train_test_split(self.datasetX, self.datasetY, test_size=holdout, random_state=random_seed)
        if(scaling == 'mean-std'):
            std_scaler = StandardScaler()
            X_train = std_scaler.fit_transform(X_train)
            X_test = std_scaler.transform(X_test)
        if(scaling =='min-max'):
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        if(data_opt == 'train'):
            self.X = torch.from_numpy(X_train)
            self.y = torch.from_numpy(y_train)
            if (oversampler == 'Random'):
                ros = RandomOverSampler(random_state=random_seed)
                self.X, self.y = ros.fit_resample(self.X, self.y)

            if (oversampler == 'ADASYN'):
                self.X, self.y = ADASYN(random_state=random_seed).fit_resample(self.X, self.y)

            if (oversampler == 'SMOTE'):
                self.X, self.y = SMOTE(random_state=random_seed).fit_resample(self.X, self.y)
        else:
            self.X = torch.from_numpy(X_test)
            self.y = torch.from_numpy(y_test)
        self.length = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.length


# Loading the Dataset into DataLoader
col_list = ['ADMISSIONS', 'ICUSTAYS', 'INPUTEVENTS_MV', 'PATIENTS']
attr_list = {'ADMISSIONS': ['DIAGNOSIS', 'ADMISSION_TYPE', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME'], 'ICUSTAYS': ['LOS'],
                                                      'INPUTEVENTS_MV': ['PATIENTWEIGHT'],'PATIENTS': ['DOB', 'GENDER']}
train_dataset = MIMIC3(col_list=col_list.copy(), attr_list=attr_list.copy())
test_dataset = MIMIC3(col_list=col_list.copy(), attr_list=attr_list.copy())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=32,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=32,
                                          shuffle=True)
input_size = 5  # X length (Must be changed)
for i, (data, labels) in enumerate(train_loader):
    input_size = data.shape[1]
    break
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
hidden_size = 20
learning_rate = 0.001

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

class FFNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FFNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        out = self.net(x)
        return out


# Train the model
def train_ffnet(model, train_loader, num_epochs):
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels.double())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Display the progress
            if (i + 1) % 300 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


# Test the model
def test_ffnet(model, test_loader):
    preds = []
    acts = []
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).double()
            outputs = model(images)
            predicted = outputs.data
            preds.extend(predicted.tolist())
            acts.extend(labels.tolist())

        preds = np.array(preds);
        acts = np.array(acts)
        yhat = torch.from_numpy(preds)
        y = torch.from_numpy(acts)
        criterion2 = RMSELoss
        loss2 = criterion2(yhat, y)
        r2 = sklearn.metrics.r2_score(acts, preds)

        # Display the result
        print("R2_Score : {}".format(r2))
        print("RMSE : {}".format(loss2))



num_classes = 1  # Y length (Must be changed)
model = (FFNet(input_size, hidden_size, num_classes).to(device)).double()

# Set the loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_ffnet(model, train_loader, num_epochs=30)
test_ffnet(model, test_loader)

