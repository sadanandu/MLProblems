# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 22:25:47 2018

@author: SHREE
"""

import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

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
        #print(each_column)
        label_encoder = LabelEncoder()
        dataframe[each_column] = label_encoder.fit_transform(dataframe[each_column])
    # categorical_features here is list of all column indexes
    # so if categorical_columns above change then this list will also
    # change
    #print(dataframe.columns)
    onehotencoder_obj = OneHotEncoder(categorical_features = [1, 2, 4, 5, 7])
    numpy_array = onehotencoder_obj.fit_transform(dataframe).toarray()
    #print(type(dataframe))
    return pandas.DataFrame(numpy_array)

file_path = "C:\\Machine Learning\\MLProblems\\Titanic Passangers\\train.csv"
dataframe = pandas.read_csv(file_path)
Y, X = get_independent_and_depedent_variables(dataframe)
#print(dataframe.isnull().sum())
X = remove_unnecessary_data(X)
X = handle_missing_data(X)
X = handle_categorical_data(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2, random_state = 7)
#print(X)

#No need of feature scaling as of now , because most of the variables are catogrical
# and they are handled. Only not categorical variables like AGE and FARE are remaining 

models = ['LogisticRegression', 'GaussianNB', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'LinearDiscriminantAnalysis' , 'SVC']
seed = 7
scoring = 'accuracy'

for each_model in models:
    classifier = globals().get(each_model)()
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(classifier, X_train, Y_train, cv =kfold, scoring = scoring)
    msg = " %f +/-(%f)" % (cv_results.mean(), cv_results.std())
    print(each_model + msg)
    #print('Accuracy score for %s : %s' % (each_model, str(accuracy_score(Y_test, Y_pred))))


#Closer to zero is better
#this is used for regressors
#print('Mean squared error- the mean value of the squared deviations of the predictions from the true values %s' % mean_squared_error(Y_test, Y_pred))

#R-squared = Explained variation / Total variation
#Higher R2 is usually better. R2 could be higher because
#of the correlation in independent variables
#  the higher the R-squared, the better the model fits your data.
#print('R2 score %s' % r2_score(Y_test, Y_pred))


