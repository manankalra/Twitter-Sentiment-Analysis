#!/usr/bin/env python

"""
SGDC
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"

import nltk
from nltk.classify.scikitlearn import SklearnClassifier  # wrapper to include the scikit-learn algorithms
from main import naive_bayes as original
from sklearn.linear_model import SGDClassifier


sgdc_classifier = SklearnClassifier(SGDClassifier())
sgdc_classifier.train(original.training_set)
sgdc_accuracy = nltk.classify.accuracy(sgdc_classifier, original.testing_set) * 100

print("\nBernoulli NB Accuracy: ", sgdc_accuracy)