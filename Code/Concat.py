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


def changeValue(datasetX):
    datasetX['LOS'] = datasetX.LOS.astype(int)
    datasetX['PATIENTWEIGHT'] = datasetX.PATIENTWEIGHT.astype(int)

    return datasetX


<<<<<<< HEAD
=======

>>>>>>> dd90a281e95f208b42f6083b37ab795546a7ba7f
def cal_age(start_time, finish_time):
    start_time = start_time.to_pydatetime()
    finish_time = finish_time.to_pydatetime()
    result = ((finish_time - start_time).days) / 365

    return result


def oneHotEncoding(data):
    enc = OneHotEncoder()
    enc.fit(data['ADMISSION_TYPE', 'GENDER'])
    data = enc.transform(data)
    return data

<<<<<<< HEAD

=======
>>>>>>> dd90a281e95f208b42f6083b37ab795546a7ba7f
def cal_days(data):
    days = []
    for i in range(0, len(data)):
        start_time = data.iloc[:, 1]
        start_time = start_time[i].to_pydatetime()
        finish_time = data.iloc[:, 3]
        finish_time = finish_time[i].to_pydatetime()
        result = (finish_time - start_time).days
        days.append(result)
    days = df(days)
    return days


class MIMIC3(torch.utils.data.Dataset):
    def __init__(self, col_list=None, attr_list=None):
        self.col_list = col_list;
        self.attr_list = attr_list
        self.col_default = ['ADMISSIONS', 'ICUSTAYS', 'INPUTEVENTS_MV', 'PATIENTS']
        self.attr_default = {'ADMISSIONS': ['ADMISSION_TYPE', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME'], 'ICUSTAYS': ['LOS'], 'INPUTEVENTS_MV': ['PATIENTWEIGHT'],
                             'PATIENTS': ['DOB', 'GENDER']}
        self.col_list = self.col_default
        self.attr_list = self.attr_default
        self.datasetX = None
        self.datasetY = None

        col_list = self.col_list
        conn = pymysql.connect(host='192.168.56.104', user='dba', password='mysql', db='mimiciiiv14', charset='utf8')
        curs = conn.cursor(pymysql.cursors.DictCursor)
        # Select LABEVENTS,SUBJECT_ID FROM LABEVENTS JOIN PATIENTS on LABEVENTS.SUBJECT_ID = PATIENTS.SUBJECT_ID JOIN ADMISSIONS on PATIENTS.SUBJECT_ID = ADMISSIONS.SUBJECT_ID
        sql_line = 'SELECT'
        for col in col_list:
            for attr in self.attr_default[col]:
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
        self.datasetX = df(result)
        self.datasetY = self.datasetX[['ADMITTIME', 'DISCHTIME', 'DEATHTIME']]
        self.datasetX = self.datasetX.drop(['ADMITTIME', 'DISCHTIME', 'DEATHTIME'], axis=1)

        self.datasetY = cal_days(df(result)).to_numpy()
        self.datasetX = changeValue(oneHotEncoding(self.datasetX)).to_numpy()

        self.X = torch.from_numpy(self.datasetX)
        self.y = torch.from_numpy(self.datasetY)
        self.length = self.datasetX.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.length


# Loading the Dataset into DataLoader
train_dataset = MIMIC3()
test_dataset = MIMIC3()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=32,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=32,
                                          shuffle=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
hidden_size = 300
learning_rate = 0.001


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
            loss = criterion(outputs, labels.long())

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
            labels = labels.to(device).long()
            outputs = model(images)
            predicted = outputs.data
            preds.extend(predicted.tolist())
            acts.extend(labels.tolist())

        preds = np.array(preds);
        acts = np.array(acts)
        r2 = sklearn.metrics.r2_score(acts, preds)

        # Display the result
        print("R2_Score : {}".format(r2))


input_size = 7  # X length (Must be changed)
num_classes = 1  # Y length (Must be changed)
model = (FFNet(input_size, hidden_size, num_classes).to(device)).double()

# Set the loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_ffnet(model, train_loader, num_epochs=100)
test_ffnet(model, test_loader)

