#library
#pip install -U imbalanced-learn(설치 방법)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#oversampling library
#from imblearn.over_sampling import *

#from imblearn.over_sampling import SMOTE, ADASYN

#choose variable from data

df = df.loc[:,["X1","X2","X3","X4", "y1"]]
df.info()
df.describe()
df.hist(bins=50, figsize=(20,15))

#make random seed
np.random.seed(42)
X = df.loc[:,["X1","X2","X3","X4"]]
y = df.loc[:, "y1"]

#from sklearn.preprocessing import MinMaxScaler
#std_scaler = MinMaxScaler()
std_scaler= StandardScaler()
X_train_std =std_scaler.fit_transform(X_train)
X_test_std =std_scaler.fit_transform(X_test)

#oversampling
'''
ros = RandomOverSampler(random_state=0)
X, y = ros.fit_resample(X, y)
'''

#ADASYN(Adaptive Synthetic Sampling) 방법은
# 소수 클래스 데이터와 그 데이터에서 가장 가까운
# k개의 소수 클래스 데이터 중 무작위로 선택된 데이터 사이의
# 직선상에 가상의 소수 클래스 데이터를 만드는 방법이다.
#X, y = ADASYN().fit_resample(X, y)

#위의 oversampling이 잘 안되면 아래꺼
#SMOTE(Synthetic Minority Over-sampling Technique) 방법도
#ADASYN 방법처럼 데이터를 생성하지만 생성된 데이터를
# 무조건 소수 클래스라고 하지 않고 분류 모형에 따라 분류한다.
#X, y = SMOTE().fit_resample(X, y)

#train test data split(70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Feature Scaling(Standardization)
#train dataset만 normalize

y_train = np.array(y_train)
y_train =y_train.reshape(-1, 1)

y_test = np.array(y_test)
y_test = y_test.reshape(-1, 1)

train = pd.concat([X_train_std, y_train], axis=1)
test = pd.concat([X_test_std, y_test], axis=1)


