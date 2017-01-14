#!/usr/bin/env python

"""
NuSVC
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"


import nltk
from nltk.classify.scikitlearn import SklearnClassifier  # wrapper to include the scikit-learn algorithms
from main import naive_bayes as original
from sklearn.svm import NuSVC


nu_svc_classifier = SklearnClassifier(NuSVC())
nu_svc_classifier.train(original.training_set)
nu_svc_accuracy = nltk.classify.accuracy(nu_svc_classifier, original.testing_set) * 100

print("\nBernoulli NB Accuracy: ", nu_svc_accuracy)
