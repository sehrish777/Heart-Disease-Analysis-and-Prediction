# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:13:15 2019

@author: M AQEEL M
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplot.pyplot as plt
%matplotlib notebook
data=pd.read_csv('')
column=['id','age','gender','height','weight','ap_hi','ao-lo','cholestrol','gluc','smoke','alco','active','cardio']
data.columns_column
data.head()
data.describe()
from sklearn.metrics import make scorer,accuracy score
from sklearn.model_selection import gridsearch.csv
from sklearn.neural_network import MLPClassifier
ann_clf=MLPClassifier()
parametres={'solver':['lbfgs'],
            'alpha':[le_4],
            'hidden_layer_sizes':(9,14,14,2),
            'random_state':[1]}
ann_scorer=make_scorer(accuracy_score)
grid_obj=GridSearchCV(ann_clf,parameters,scoring=acc_scorer)
grid_obj=grid_obj.fit(X_train,Y_train)
ann_clf=grid_obj.best_estimator
ann_clf.fit(X_train,Y_train)
y_pred_ann=ann_clf_predict(X_test)
from sklearn.metrics import confusion_matrix
cm_ann=confusion_matrix(Y_test,Y_pred_ann)
cm_ann
ann_result=accuracy_score(Y-test,Y_pred_ann)
ann_result