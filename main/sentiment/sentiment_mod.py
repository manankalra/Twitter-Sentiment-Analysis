#!/usr/bin/env python

"""
Main function that tags the input data-text as positive or negative sentiment; trained on a large and custom data-set
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"


import nltk
import random
import pickle

from nltk.classify import ClassifierI

from statistics import mode

from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# # training-data
short_pos = open("short_reviews\positive.txt", "r").read()
short_neg = open("short_reviews\\negative.txt", "r").read()
documents, all_words = [], []

# J is adjective, R is adverb, and V is verb
# allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in short_pos.split("\n"):
    documents.append((p, "pos"))
    words = word_tokenize(p)
    short_pos_words = nltk.pos_tag(words)
    for w in short_pos_words:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for n in short_neg.split("\n"):
    documents.append((n, "neg"))
    words = word_tokenize(n)
    short_neg_words = nltk.pos_tag(words)
    for w in short_neg_words:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

documents_temp = open("pickled_algorithms/documents.pickle", "rb")
documents = pickle.load(documents_temp)
documents_temp.close()

word_features_temp = open("pickled_algorithms/word_features.pickle", "rb")
word_features = pickle.load(word_features_temp)
word_features_temp.close()

# # returns a dictionary, keys = word_features, value =  Boolean (acc. to the presence of that feature in a  document)
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

feature_sets_temp = open("pickled_algorithms/feature_sets.pickle", "rb")
feature_sets = pickle.load(feature_sets_temp)
feature_sets_temp.close()

random.shuffle(feature_sets)
print(len(feature_sets))

training_set = feature_sets[:10000]
testing_set = feature_sets[10000:]


########################################################################################################################
open_file = open("pickled_algorithms/original_naive_bayes.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algorithms/multinomial_naive_bayes.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algorithms/bernoulli_naive_bayes.pickle", "rb")
Bern_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algorithms/logistic_regression.pickle", "rb")
logistic_regression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algorithms/sgdc.pickle", "rb")
sgdc_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algorithms/svc.pickle", "rb")
svc_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algorithms/linear_svc.pickle", "rb")
linear_svc_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algorithms/nu_svc.pickle", "rb")
nu_svc_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  Bern_classifier,
                                  logistic_regression_classifier,
                                  sgdc_classifier,
                                  linear_svc_classifier,
                                  nu_svc_classifier)


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats) * 100
########################################################################################################################