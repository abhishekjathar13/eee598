# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 06:50:29 2018

@author: Abhishek
"""

import pandas as pd

 

dataset = pd.read_table( r'C:\Users\Abhishek\Desktop\Machine Learning\Project\Jathar_project_final.txt')

 

feature_names = ['Elec_Facility',

                 'Elec_Fans',

                 'Elec_Cooling',

                 'Elec_Heating',

                 'Elec_InteriorLights',

                 'Elec_InteriorEquipment',

                 'Gas_Facility',

                 'Gas_Heating',

                 'Gas_InteriorEquipment',

                 'Gas_Water Heater']

X = dataset[feature_names].astype(float)

y = dataset['Site']

 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

 

#Logistic Regression

 

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'

     .format(logreg.score(X_train, y_train)))

print('Accuracy of Logistic regression classifier on test set: {:.2f}'

     .format(logreg.score(X_test, y_test)))

 

# Naive Bayes

 

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

print('Accuracy of GNB classifier on training set: {:.2f}'

     .format(gnb.score(X_train, y_train)))

print('Accuracy of GNB classifier on test set: {:.2f}'

     .format(gnb.score(X_test, y_test)))

 

# SVM

 

from sklearn.svm import SVC

svm = SVC()

svm.fit(X_train, y_train)

print('Accuracy of SVM classifier on training set: {:.2f}'

     .format(svm.score(X_train, y_train)))

print('Accuracy of SVM classifier on test set: {:.2f}'

     .format(svm.score(X_test, y_test)))

 

# Neural Network

 

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,

                    hidden_layer_sizes=(5, 2), random_state=1)

#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,

#                   hidden_layer_sizes=(15,), random_state=1)

clf.fit(X_train, y_train)                                  

print('Accuracy of Neural Network classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))

print('Accuracy of Neural Network classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))

from keras.utils.vis_utils import plot_model
