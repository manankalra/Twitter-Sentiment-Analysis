#!/usr/bin/env python

"""
NaiveBayes
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"


import nltk
import random
from nltk.corpus import movie_reviews


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

# #  Naive-Bayes (posterior = (prior occurrences * likelihood) / evidence)
classifier = nltk.NaiveBayesClassifier.train(training_set)
accuracy = nltk.classify.accuracy(classifier, testing_set) * 100

print("Original NB Accuracy: ", accuracy)
classifier.show_most_informative_features()

'''
# save
save_classifier = open("naivebayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# load
temp = open("naivebayes.pickle", "rb")
myclassifier = pickle.load(temp)
myclassifier.close()
'''

