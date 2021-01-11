# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:38:54 2021

@author: isscmi
"""


import spacy
import os
import numpy as np
import pandas as pd
import xmltodict
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_PATH = "d:\github\speech\Bundesregierung.xml"
nlp = spacy.load('de_core_news_sm')


def open_data():
    file = open(DATA_PATH, mode="rb")
    xml_document = xmltodict.parse(file)
    text_nodes = xml_document['collection']['text']
    df = pd.DataFrame({'person': [t['@person'] for t in text_nodes],
                       'speech': [t['rohtext'] for t in text_nodes]})
    df["length"] = df.speech.str.len()

    sns.countplot(y="person", data=df).set(title="Anzahl an Reden", xlabel='',
                                           ylabel='')

    sns.boxplot(y="person", x="length", data=df).\
        set(title="Länge der Reden in Zeichen", xlabel="", ylabel="")
    # remove speeches wothout names
    df = df[df['person'] != 'k.A.']
    # only person with more then 50 speeches, select 50 of them
    df = df.groupby('person').apply\
        (lambda g: g.sample(0 if len(g) < 50else 50)).reset_index(drop=True)
    df['analysis'] = df.speech.map(analyze)
    df['tokens'] = df.analysis.apply(lambda x: x[0])
    df['lemmata'] = df.analysis.apply(lambda x: x[1])

    return df


def analyze(speech):
    # POS-Tagging and Parsong off, run faster
    with nlp.disable_pipes('tagger', 'parser'):
        document = nlp(speech)
        token = [w.text for w in document]
        lemma = [w.lemma_ for w in document]

        return (token, lemma)


def train_test_evaluate(speeches, persons):
    # Durchnummerieren der Redner
    encoder = LabelEncoder()
    y = encoder.fit_transform(persons)
    # Bag of Words der Reden extrahieren
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(speeches).toarray()
    # Daten aufteilen für Training und Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Klassifikator trainieren und testen
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Vorhersage-Genauigkeit auswerten
    print(accuracy_score(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_)


if __name__ == '__main__':
    df = open_data()
    train_test_evaluate(df['speech'], df['person'])
