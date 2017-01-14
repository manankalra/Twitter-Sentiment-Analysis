#!/usr/bin/env python

"""
Lemmatizing
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"


from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("better", pos='a'))
