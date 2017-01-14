#!/usr/bin/env python

"""
movie_reviews (Finding accuracy of each algorithm)
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"



import nltk
import random
import pickle

from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

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


# # counting the frequency of each word
all_words = nltk.FreqDist(all_words)  # <FreqDist with 39768 samples and 1583820 outcomes>
# print(all_words.most_common(15))
# print(all_words["<random_word>"])

# # list of all the words (non-repeated), top 5000
word_features = list(all_words.keys())[:5000]

# # saving trained data
save_word_features = open("pickled_algorithms\word_features.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


# # returns a dictionary, keys = word_features, value =  Boolean (acc. to the presence of that feature in a  document)
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

# # transforming documents
feature_sets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(feature_sets)

training_set = feature_sets[:10000]
testing_set = feature_sets[10000:]


########################################################################################################################
# #  Naive-Bayes (posterior = (prior occurrences * likelihood) / evidence)
classifier = nltk.NaiveBayesClassifier.train(training_set)
accuracy = nltk.classify.accuracy(classifier, testing_set) * 100

print("Original NB Accuracy: ", accuracy)
classifier.show_most_informative_features()

save_classifier = open("pickled_algorithms\original_naive_bayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


# # Multinomial Naive-Bayes
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
multi_accuracy = nltk.classify.accuracy(MNB_classifier, testing_set) * 100

print("\nMultinomial NB Accuracy: ", multi_accuracy)

save_classifier = open("pickled_algorithms\multinomial_naive_bayes.pickle", "wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

# # Bernoulli Naive-Bayes
Bern_classifier = SklearnClassifier(BernoulliNB())
Bern_classifier.train(training_set)
bern_accuracy = nltk.classify.accuracy(Bern_classifier, testing_set) * 100

print("\nBernoulli NB Accuracy: ", bern_accuracy)

save_classifier = open("pickled_algorithms\\bernoulli_naive_bayes.pickle", "wb")
pickle.dump(Bern_classifier, save_classifier)
save_classifier.close()

# # Logistic Regression
logistic_regression_classifier = SklearnClassifier(LogisticRegression())
logistic_regression_classifier.train(training_set)
log_accuracy = nltk.classify.accuracy(logistic_regression_classifier, testing_set) * 100

print("\nLogistic Regression Accuracy: ", log_accuracy)

save_classifier = open("pickled_algorithms\logistic_regression.pickle", "wb")
pickle.dump(logistic_regression_classifier, save_classifier)
save_classifier.close()

# # SGDC
sgdc_classifier = SklearnClassifier(SGDClassifier())
sgdc_classifier.train(training_set)
sgdc_accuracy = nltk.classify.accuracy(sgdc_classifier, testing_set) * 100

print("\nSGDC Accuracy: ", sgdc_accuracy)

save_classifier = open("pickled_algorithms\sgdc.pickle", "wb")
pickle.dump(sgdc_classifier, save_classifier)
save_classifier.close()

# # SVC
svc_classifier = SklearnClassifier(SVC())
svc_classifier.train(training_set)
svc_accuracy = nltk.classify.accuracy(svc_classifier, testing_set) * 100

print("\nSVC Accuracy: ", svc_accuracy)

save_classifier = open("pickled_algorithms\svc.pickle", "wb")
pickle.dump(svc_classifier, save_classifier)
save_classifier.close()


# # Linear SVC
linear_svc_classifier = SklearnClassifier(LinearSVC())
linear_svc_classifier.train(training_set)
linear_svc_accuracy = nltk.classify.accuracy(linear_svc_classifier, testing_set) * 100

print("\nLinear SVC Accuracy: ", linear_svc_accuracy)

save_classifier = open("pickled_algorithms\linear_svc.pickle", "wb")
pickle.dump(linear_svc_classifier, save_classifier)
save_classifier.close()

# # Nu SVC
nu_svc_classifier = SklearnClassifier(NuSVC())
nu_svc_classifier.train(training_set)
nu_svc_accuracy = nltk.classify.accuracy(nu_svc_classifier, testing_set) * 100

print("\nNu SVC Accuracy: ", nu_svc_accuracy)

save_classifier = open("pickled_algorithms\\nu_svc.pickle", "wb")
pickle.dump(nu_svc_classifier, save_classifier)
save_classifier.close()

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