import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
#import mpld3
import matplotlib.pyplot as plt
import matplotlib as mpl


df = pd.read_csv(r"data.csv")
df = df['keyword']

i = 0
for line in df:
    line = re.sub(r'(\<(/?[^>]+)>)', ' ', line)
    #chis = re.sub()
    line = re.sub('[^а-яА-Я ]', '', line)
    df[i] = line
    #posti.append(chis)
    #posti[i] = chis
    i += 1

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("russian")

def token_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

token_and_stem('sad')

def token_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    return filtered_tokens

#Создаем словари (массивы) из полученных основ
totalvocab_stem = []
totalvocab_token = []
for i in df:
    allwords_stemmed = token_and_stem(i)
    #print(allwords_stemmed)
    totalvocab_stem.extend(allwords_stemmed)
    
    allwords_tokenized = token_only(i)
    totalvocab_token.extend(allwords_tokenized)

stopwords = nltk.corpus.stopwords.words('russian')
#можно расширить список стоп-слов
stopwords.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'на'])

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

n_featur=200000
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000,
                                 min_df=0.01, stop_words=stopwords,
                                 use_idf=True, tokenizer=token_and_stem, ngram_range=(1,3))
from IPython import get_ipython
ipython = get_ipython()

tfidf_matrix = tfidf_vectorizer.fit_transform(df)
#ipython.magic('time tfidf_matrix = tfidf_vectorizer.fit_transform(titles)')
#print(tfidf_matrix.shape)

num_clusters = 5

# Метод к-средних - KMeans
from sklearn.cluster import KMeans

km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
idx = km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

#print(clusters)
#print (km.labels_)

#k-means
clusterkm = km.labels_.tolist()

df1 = df.tolist()
frame = pd.DataFrame(df1, index = [clusterkm])

#k-means
out = { 'title': df1, 'cluster': clusterkm }
resultDF = pd.DataFrame(out, columns = ['title', 'cluster'])
#print(resultDF)
resultDF.to_csv("result.txt", sep='\t', encoding='utf-8')

'''
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
dist.shape
'''

