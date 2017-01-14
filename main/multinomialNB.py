#!/usr/bin/env python

"""
MultinomialNaiveBayes
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"


import nltk
from nltk.classify.scikitlearn import SklearnClassifier  # wrapper to include the scikit-learn algorithms within the nltk classifier
from sklearn.naive_bayes import MultinomialNB
from main import naive_bayes as original


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(original.training_set)
multi_accuracy = nltk.classify.accuracy(MNB_classifier, original.testing_set) * 100

print("\nMultinomial NB Accuracy: ", multi_accuracy)
