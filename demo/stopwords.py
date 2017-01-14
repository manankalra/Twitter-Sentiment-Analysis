#!/usr/bin/env python

"""
Eliminating Stop-Words
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


sample = "Hey! NLTK is a Python toolkit for Natural Language Processing. Do try it out."

stop_words = set(stopwords.words("english"))
print(stop_words)

toPrint = [w for w in word_tokenize(sample) if w not in stop_words]
print(toPrint)