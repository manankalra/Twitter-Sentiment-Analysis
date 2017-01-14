#!/usr/bin/env python

"""
Wordnet
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"


from nltk.corpus import wordnet


syns = wordnet.synsets("good")

# synset
print(syns)
print()


# just the word, (lemmas are like synonyms/antonyms)
print(syns[0].lemmas()[0].name())
print()


# definition
print(syns[0].definition())
print()


# examples
print(syns[0].examples())
print()


# synonyms and antonyms
synonyms = []
antonyms = []
for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        print("lemma: ", l, end=' ')
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.name())

print("Synonyms: ", set(synonyms))
print("Antonyms: ", set(antonyms))
print()


# similarity in %
w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("dog.n.01")
print(w1.wup_similarity(w2))
print()

