#!/usr/bin/env python

"""
Logisticregression
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"


import nltk
from nltk.classify.scikitlearn import SklearnClassifier  # wrapper to include the scikit-learn algorithms
from main import naive_bayes as original
from sklearn.linear_model import LogisticRegression


logistic_regression_classifier = SklearnClassifier(LogisticRegression())
logistic_regression_classifier.train(original.training_set)
log_accuracy = nltk.classify.accuracy(logistic_regression_classifier, original.testing_set) * 100

print("\nLogistic Regression Accuracy: ", log_accuracy)
