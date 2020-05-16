# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:22:38 2020

@author: Lenovo
"""

import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix

df= pd.read_csv('parkinsons.data')
df.head()

features = df.drop('status',axis='columns')
features = features.iloc[:,1:]

#features1  = df.loc[:,df.columns!='status'].values[:,1:]
#labels=df.loc[:,'status'].values
labels= df['status']

labels.value_counts()

#print(labels[labels==1].shape[0], labels[labels==0].shape[0])


scaler=MinMaxScaler((-1,1))

x=scaler.fit_transform(features)
y=labels

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=0)

xgb=XGBClassifier()
xgb.fit(x_train,y_train)

from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression()
lgr.fit(x_train,y_train)

#from sklearn.svm import SVC
from sklearn import svm
svm1 = svm.SVC(kernel='linear',C=1 ,probability = True)
svm1.fit(x_train,y_train)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=400,random_state=1)
rfc.fit(x_train,y_train)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB().fit(x_train,y_train)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier().fit(x_train,y_train)


xgb_pred=xgb.predict(x_test)
lgr_pred = lgr.predict(x_test)
svm_pred = svm1.predict(x_test)
rfc_pred = rfc.predict(x_test)
nb_pred = nb.predict(x_test)
dtc_pred = dtc.predict(x_test)


print("Accuracy of XGBOOST:",accuracy_score(y_test, xgb_pred)*100)

print("Accuracy of LogisticRegression:",accuracy_score(y_test, lgr_pred)*100)

print("Accuracy of  SVM:",accuracy_score(y_test, svm_pred)*100)

print("Accuracy of RandomForestClassifier:",accuracy_score(y_test, rfc_pred)*100)

print("Accuracy of  Naive Bayes:",accuracy_score(y_test, nb_pred)*100)

print("Accuracy of  Decision Tree Classifier:",accuracy_score(y_test, dtc_pred)*100)




print(confusion_matrix(y_test,xgb_pred))












