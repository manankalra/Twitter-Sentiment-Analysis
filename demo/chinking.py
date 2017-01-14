#!/usr/bin/env python

"""
Chinking
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"


import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


sample_text = state_union.raw("sample.txt")

custom_sent_tokenizer = PunktSentenceTokenizer()
tokenized = custom_sent_tokenizer.tokenize(sample_text)


def tag():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunk = r"""Chunk: {<.*>+}
                                     } <VB.?|IN|DT>+{"""
            chunk_parser = nltk.RegexpParser(chunk)
            chunked = chunk_parser.parse(tagged)
            chunked.draw()
    except Exception as e:
        print(str(e))


tag()

