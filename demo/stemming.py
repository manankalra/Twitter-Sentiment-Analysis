#!/usr/bin/env python

"""
Stemming
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"



from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

sample = "I'm learning Python. Trying to write pythonic code. Pythonically, trying stemming using NLTK."
ps = PorterStemmer()

for w in word_tokenize(sample):
    print(ps.stem(w), end=' ')
