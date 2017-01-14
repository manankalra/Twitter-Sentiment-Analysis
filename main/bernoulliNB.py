#!/usr/bin/env python

"""
BernoulliNaiveBayes
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"


import nltk
from nltk.classify.scikitlearn import SklearnClassifier  # wrapper to include the scikit-learn algorithms
from sklearn.naive_bayes import BernoulliNB
from main import naive_bayes as original


Bern_classifier = SklearnClassifier(BernoulliNB())
Bern_classifier.train(original.training_set)
bern_accuracy = nltk.classify.accuracy(Bern_classifier, original.testing_set) * 100

print("\nBernoulli NB Accuracy: ", bern_accuracy)
