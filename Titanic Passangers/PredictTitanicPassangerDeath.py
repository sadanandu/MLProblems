# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 22:25:47 2018

@author: SHREE
"""

import pandas
import numpy
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer


# TransformerMixin is base class for all sklearn transformers
from sklearn.base import TransformerMixin

class CategoricalImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Imputed with the most frequent value 
        in columns. This only works on Series
        """
    def fit(self, X, y=None):

        self.fill = X.value_counts().index[0] 

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

def fill_missing_values_with_mean(series):
    return series.fillna(series.mean())

def fill_missing_values_with_imputer(dataframe):
    #Tried many ways with standard Imputer object but it always gives an error
    # ValueError: could not convert string to float: 'Q'
    # This is because Imputer's fit and transform both work on float types.
    '''#print(series[61])
    #print(type(series[829]))
    #series = series.fillna(numpy.NaN)
    #print(series.where(series.isnull() == True))
    #print(series)
    print(type(dataframe['Embarked'][61]))
    series = dataframe['Embarked']
    imputer = Imputer()#missing_values='', strategy='most_frequent', axis=1)
    series = imputer.fit_transform(series.values)
    print(series)    
    dataframe['Embarked'] = series#  imputer.transform(series)
    #return series'''
    dataframe['Embarked'] = CategoricalImputer().fit_transform(dataframe['Embarked'])
    
    return dataframe


def handle_missing_data(dataframe):
    #Age and Cabin columns have lot of missing data
    # Age can be filled with mean data
    
    dataframe['Age'] = fill_missing_values_with_mean(dataframe['Age'])
    #print(dataframe.isnull().sum())
    dataframe = fill_missing_values_with_imputer(dataframe)

    
    return dataframe

def remove_unnecessary_data(dataframe):
    # Cabin cant be filled for missing data ...so dropping it
    # axis =1 for columns
    # axis = 0 for rows
    dataframe = dataframe.drop('Cabin', axis = 1)
    
    #there seems no use of column Tickets so dropping it
    dataframe = dataframe.drop('Ticket', axis = 1)
    
    #Removing name column it is not useful for now and making lot of problems
    dataframe = dataframe.drop('Name', axis = 1)
    return dataframe

def get_independent_and_depedent_variables(dataframe):
    return dataframe['Survived'], dataframe.drop('Survived', axis=1)

def handle_categorical_data(dataframe):
    categorical_columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
    
    for each_column in categorical_columns:
        print(each_column)
        label_encoder = LabelEncoder()
        dataframe[each_column] = label_encoder.fit_transform(dataframe[each_column])
    # categorical_features here is list of all column indexes
    # so if categorical_columns above change then this list will also
    # change
    print(dataframe.columns)
    onehotencoder_obj = OneHotEncoder(categorical_features = [1, 2, 4, 5, 7])
    numpy_array = onehotencoder_obj.fit_transform(dataframe).toarray()
    #print(type(dataframe))
    return pandas.DataFrame(numpy_array)

def main():
    file_path = "C:\\Machine Learning\\MLProblems\\Titanic Passangers\\train.csv"
    dataframe = pandas.read_csv(file_path)
    Y, X = get_independent_and_depedent_variables(dataframe)
    #print(dataframe.isnull().sum())
    X = remove_unnecessary_data(X)
    X = handle_missing_data(X)
    X = handle_categorical_data(X)
    print(X.columns)
    #print(X)

main()