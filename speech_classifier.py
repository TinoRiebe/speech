# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:38:54 2021

@author: isscmi
"""


import spacy
import os
import numpy as np
import pandas as pd
import urllib
import zipfile
import xmltodict
import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = "d:\github\speech\Bundesregierung.xml"


with open(DATA_PATH, mode="rb") as file:
    xml_document = xmltodict.parse(file)
    text_nodes = xml_document['collection']['text']
    df = pd.DataFrame({'person' : [t['@person'] for t in text_nodes],
                        'speech' : [t['rohtext'] for t in text_nodes]})


config InlineBackend.figure_format = 'svg' # schönere Grafiken
sns.set()

df["length"] = df.speech.str.len()
sns.countplot(y="person", data=df).set(title="Anzahl an Reden", xlabel='', ylabel='')
sns.boxplot(y="person", x="length", data=df).set(title="Länge der Reden in Zeichen", xlabel="", ylabel="")

# reden ohne namenzuweisung löschen

df = df[df['person']!= 'k.A.']

df1 = df.groupby('person').apply(lambda g: g.sample(0 if len(g) < 50 else 50)).reset_index(drop=True)

df1
# POS-Tagging und Parsing aus, zwecks schneller
def analyze(speech):
    with nlp.disable_pipes('tagger', 'parser'):
        document = nlp(speech)
        token = [w.text for w in document]
        lemma = [w.lemma_ for w in document]
        
        return (token, lemma)

df['analysis'] = df.speech.map(analyze)
df['tokens'] = df.analysis.apply(lambda x: x[0])
df['lemmata'] = df.analysis.apply(lambda x: x[1])

def BagOfWords(speeches):
    word_sets = [set(speech) for speech in speeches]
    vocabulary = list(set.union(*word_sets))
    set2bow = lambda s: [1 if w in s else o for w in vocabulary]
    return (vocabulary, list(map(set2bow, word_sets)))

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

def train_test_evaluate(speeches, persons):
    # Durchnummerieren der Redner
    encoder = LabelEncoder()
    y = encoder.fit_transform(persons)
    # Bag of Words der Reden extrahieren
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(speeches).toarray()
    # Daten aufteilen für Training und Test
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
    # Klassifikator trainieren und testen
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Vorhersage-Genauigkeit auswerten
    print(accuracy_score(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_)
    

train_test_evaluate(df1['speech'], df1['person'])
