#!/usr/bin/env python

"""
LinearSVC
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"


import nltk
from nltk.classify.scikitlearn import SklearnClassifier  # wrapper to include the scikit-learn algorithms
from main import naive_bayes as original
from sklearn.svm import LinearSVC


linear_svc_classifier = SklearnClassifier(LinearSVC())
linear_svc_classifier.train(original.training_set)
linear_svc_accuracy = nltk.classify.accuracy(linear_svc_classifier, original.testing_set) * 100

print("\nBernoulli NB Accuracy: ", linear_svc_accuracy)
