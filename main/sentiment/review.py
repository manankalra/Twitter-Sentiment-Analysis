#!/usr/bin/env python

"""
movie_reviews (temp)
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"


import nltk
import random

from nltk.corpus import movie_reviews

from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI

from statistics import mode


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
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# # list of all the words; like everything in our data-set
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

# # counting the frequency of each word
all_words = nltk.FreqDist(all_words)  # <FreqDist with 39768 samples and 1583820 outcomes>
# print(all_words.most_common(15))
# print(all_words["<random_word>"])

# # list of all the words (non-repeated), top 3000
word_features = list(all_words.keys())[:3000]


# # returns a dictionary, keys = word_features, value =  Boolean (acc. to the presence of that feature in a  document)
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

# # transforming documents
feature_sets = [(find_features(rev), category) for (rev, category) in documents]

training_set = feature_sets[:1900]
testing_set = feature_sets[1900:]


########################################################################################################################
# #  Naive-Bayes (posterior = (prior occurrences * likelihood) / evidence)
classifier = nltk.NaiveBayesClassifier.train(training_set)
accuracy = nltk.classify.accuracy(classifier, testing_set) * 100

print("Original NB Accuracy: ", accuracy)
classifier.show_most_informative_features()

# # Multinomial Naive-Bayes
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
multi_accuracy = nltk.classify.accuracy(MNB_classifier, testing_set) * 100

print("\nMultinomial NB Accuracy: ", multi_accuracy)

# # Bernoulli Naive-Bayes
Bern_classifier = SklearnClassifier(BernoulliNB())
Bern_classifier.train(training_set)
bern_accuracy = nltk.classify.accuracy(Bern_classifier, testing_set) * 100

print("\nBernoulli NB Accuracy: ", bern_accuracy)

# # Logistic Regression
logistic_regression_classifier = SklearnClassifier(LogisticRegression())
logistic_regression_classifier.train(training_set)
log_accuracy = nltk.classify.accuracy(logistic_regression_classifier, testing_set) * 100

print("\nLogistic Regression Accuracy: ", log_accuracy)

# # SGDC
sgdc_classifier = SklearnClassifier(SGDClassifier())
sgdc_classifier.train(training_set)
sgdc_accuracy = nltk.classify.accuracy(sgdc_classifier, testing_set) * 100

print("\nSGDC Accuracy: ", sgdc_accuracy)

# # SVC
svc_classifier = SklearnClassifier(SVC())
svc_classifier.train(training_set)
svc_accuracy = nltk.classify.accuracy(svc_classifier, testing_set) * 100

print("\nSVC Accuracy: ", svc_accuracy)


# # Linear SVC
linear_svc_classifier = SklearnClassifier(LinearSVC())
linear_svc_classifier.train(training_set)
linear_svc_accuracy = nltk.classify.accuracy(linear_svc_classifier, testing_set) * 100

print("\nLinear SVC Accuracy: ", linear_svc_accuracy)

# # Nu SVC
nu_svc_classifier = SklearnClassifier(NuSVC())
nu_svc_classifier.train(training_set)
nu_svc_accuracy = nltk.classify.accuracy(nu_svc_classifier, testing_set) * 100

print("\nNu SVC Accuracy: ", nu_svc_accuracy)

# # Voting # #
voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  Bern_classifier,
                                  logistic_regression_classifier,
                                  sgdc_classifier,
                                  linear_svc_classifier,
                                  nu_svc_classifier)
voted_accuracy = nltk.classify.accuracy(voted_classifier, testing_set) * 100
print("\nVoted Accuracy: ", voted_accuracy)
# print("\nClassification:", voted_classifier.classify(testing_set[0][0]),
# "Confidence:", voted_classifier.confidence(testing_set[0][0]))
########################################################################################################################

