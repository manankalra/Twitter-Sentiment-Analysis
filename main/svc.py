#!/usr/bin/env python

"""
SVC
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"


import nltk
from nltk.classify.scikitlearn import SklearnClassifier  # wrapper to include the scikit-learn algorithms
from main import naive_bayes as original
from sklearn.svm import SVC

svc_classifier = SklearnClassifier(SVC())
svc_classifier.train(original.training_set)
svc_accuracy = nltk.classify.accuracy(svc_classifier, original.testing_set) * 100

print("\nBernoulli NB Accuracy: ", svc_accuracy)
