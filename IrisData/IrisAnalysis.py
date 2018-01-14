# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 14:56:16 2018

@author: SHREE
"""
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

file_path = "C:\\Machine Learning\\IrisData\\iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(file_path, names=names)


#shape
print(dataset.shape)

#head
print(dataset.head(20))

#descriptions
print(dataset.describe())

#class distribution
#gives number of records belonging to each group
print(dataset.groupby('class').size())

#box and whisker plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#histograms
dataset.hist()
plt.show()

#scatter plots
scatter_matrix(dataset)
plt.show()

#create train and predict datasets by splitting dataset
# we will predict class of the flower from measurements of its sepal and petals
# take all rows with lengths into X and allrows with class into Y
# to create training and predict datasets we will divide 80:20
X = dataset.values[:, 0:4]
Y = dataset.values[:, 4]

validation_size = .2
seed = 7

x_train, x_predict, y_train, y_predict = model_selection.train_test_split(X, Y, test_size= validation_size, random_state=seed)


#Create models and do k-fold cross validation.
# this means compare performance of model k times 
seed = 7
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv =kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


#Compare algorithms by plotting them
'''fig = plt.figure()
fig.suptitle("Comaprison of models")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()'''


#Using KNN model as it was most accurate
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
p = knn.predict(x_predict)
print(accuracy_score(y_predict, p))
print(confusion_matrix(y_predict, p))
print(classification_report(y_predict, p))

