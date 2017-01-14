#!/usr/bin/env python

"""
GaussianNaiveBayes (MIGHT NOT WORK; Don't use it for classifying)
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"


import nltk
from nltk.classify.scikitlearn import SklearnClassifier  # wrapper to include the scikit-learn algorithms within the nltk classifier
from sklearn.naive_bayes import GaussianNB
from main import naive_bayes as original


GSN_classifier = SklearnClassifier(GaussianNB())
GSN_classifier.train(original.training_set)
gauss_accuracy = nltk.classify.accuracy(GSN_classifier, original.testing_set) * 100

print("\nGaussian NB Accuracy: ", gauss_accuracy)
