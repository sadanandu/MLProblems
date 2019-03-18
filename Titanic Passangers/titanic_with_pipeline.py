# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:59:56 2019

@author: e1077783
"""

import sklearn.datasets
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def required_columns(df):
    col_names = ['Survived', 'Name', 'PassengerId', 'Ticket', 'Cabin', 'SibSp', 'Parch']
    req_cols = []
    for each in df.columns:
        if each not in col_names:
            req_cols.append(each)
    return req_cols

titanic = pd.read_csv("train.csv")
X , Y = titanic.loc[:, required_columns],  titanic.loc[:, titanic.columns == 'Survived']
X_train,X_test, Y_train,Y_test = train_test_split(X,Y, random_state=0)

col_with_nan = ['Age']
col_to_scale = ['Fare']
col_to_encode = ['Embarked', 'Sex', 'Pclass']#[0,1,4,5]

imputer = Pipeline(steps = [ ('imp', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
scaler = Pipeline(steps = [('scaler', StandardScaler())])
one_hot_encoder = Pipeline(steps = [('imp2', SimpleImputer(strategy='most_frequent')), 
                                         ('onhot_enc', OneHotEncoder(handle_unknown='ignore')) ])
scaler = Pipeline(steps = [('scale', StandardScaler())])
    
transformer = ColumnTransformer(
        transformers = [('imp_scale', imputer, col_with_nan), \
                        ('scaler', scaler, col_to_scale), \
                        ('imp_enc_only', one_hot_encoder, col_to_encode)],
        #sparse_threshold = 0,
        remainder = 'passthrough'
        )  
#x_tranformed = transformer.fit_transform(X_train, Y_train)
label_encoder = LabelEncoder()
label_encoder.fit_transform(Y_train)  

pipeline = Pipeline(steps=[('transformer', transformer), ('classified', LogisticRegression(solver='lbfgs'))])
pipeline.fit(X_train, Y_train)

print(pipeline.score(X_test,Y_test))