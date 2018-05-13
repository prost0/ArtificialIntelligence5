Лабораторная работа №5
==
Вариант 3. Реализовать алгоритм выявляющий взаимосвязанные сообщения на языке Python. Подобрать или создать датасет и обучить модель. Продемонстрировать зависимость качества кластеризации от объема, качества выборки и числа кластеров. Продемонстрировать работу вашего алгоритма. Обосновать выбор данного алгоритма машинного обучения. Построить облако слов для центров кластеров(wordcloud).

Загрузим необходимые пакеты:
```python
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import matplotlib.pyplot as plt
import matplotlib as mpl
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
```

Для своей работы я выбрал метод k-means (метод k средних) из-за его простоты, высокой скорости выполнения, а также он требует задания числа кластеров, что необходимо для выполнения задачи(продемонстрировать зависимость качества кластеризации от числа кластеров).

ai5.py
==
Загрузим датасет с поисковыми запросами data.txt, оставим только нужные нам данные:
```python
df = pd.read_csv(r"data.csv")
df = df['keyword']
```
Удалим все знаки препинания и цифры:
```python
i = 0
for line in df:
    line = re.sub(r'(\<(/?[^>]+)>)', ' ', line)
    line = re.sub('[^а-яА-Я ]', '', line)
    df[i] = line
    i += 1
```
Приступаем к нормализации – приведению слова к начальной форме с помощью стеммера Портера:
```python
def token_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

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
```

Создадим матрицу весов TF-IDF. Будем считать каждый поисковой запрос за документ. tfidf_vectorizer мы возьмем из пакета sklearn, а стоп-слова мы возьмем из корпуса ntlk:
```python
stopwords = nltk.corpus.stopwords.words('russian')
#расширим список стоп-слов
stopwords.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'на'])
n_featur=200000
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000,
                                 min_df=0.01, stop_words=stopwords,
                                 use_idf=True, tokenizer=token_and_stem, ngram_range=(1,3))
```
Над полученной матрицей применим метод кластеризации k-means:
```python
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
idx = km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
```
Сохраним полученные данные в файл result.txt:
```python
df1 = df.tolist()
frame = pd.DataFrame(df1, index = [clusterkm])
out = { 'title': df1, 'cluster': clusterkm }
resultDF = pd.DataFrame(out, columns = ['title', 'cluster'])
resultDF.to_csv("result.txt", sep='\t', encoding='utf-8')
```

Перед следующим пунктом стоит сказать, что задача кластеризации не имеет точного решения и критерии оценки качества сугубо субъективны.
Зависимость качества кластеризации от числа кластеров
==

Посмотрим на результаты кластеризации для различного количества кластеров:

1. 5 кластеров
```
    3934	новослободский дом интернат для престарелых и инвалидов	0
    3940	номер гос контракта мкм логистика	0
    3956	обезьянский денис сергеевич	0
    3775	мбу парки города	0
```
Как видим, абсолютно непохожие запросы находятся в одном кластере, попробуем увеличить количество кластеров.

2. 10 клсатеров

```
3765	мбу владстройзаказчик владимир официальный сайт	7
3865	мку центр гражданской защиты города ярославля официальный сайт	7
3946	оао газпром газораспределение элиста официальный сайт	7
4193	ростехнадзор североуральское управление тюмень официальный сайт	7
4202	сайт гос затраты	7
4565	фку гб мсэ по архангельской области минтруда россии	9
4644	форум фку налогсервис	9
4402	ухтинский вгсофилиал фку асф свпфвч	9
4571	фку гб мсэ по рт	9
4704	энергосбыт ростовэнерго каменскшахтинский щаденко	0
4705	эсвком вологда добролюбова	0
4706	ювент компани	0
4707	юзмо оао энергосбыт ростовэнерго таганрог	0
4708	юлмарт региональная сбытовая компания	0
```

Видим, что результаты значительно улучшились, однако запросы различной тематитики все ещё могут попасть в один кластер.

3. 15 кластеров

```
4789	 нии мо рф официальный сайт	11
4919	соколинский психоневрологический интернат официальный сайт	11
4935	агентство по муниципальному заказу жкх ярославль официальный сайт	11
4942	административный департамент ивановской области	10
4950	администрация города ясный оренбургской области	10
4978	администрация ровеньки белгородской области	10
5417	гбоу впо нгму минздрава россии	12
5418	гбоу впо нижгма минздрава россии	12
5419	гбоу впо ростгму минздрава россии ростов на дону	12
5841	госзатраты ооо проект м	14
6527	краснодартранспроект ооо	14
7365	ооо строй сервис проект	14
```
Наконец мы получили достойный результат, увеличивать количество кластеров дальше не имеет смысла, так как тогда похожие запросы будут находиться в разных кластерах.

Стоит отметить, что выбор количества кластеров всегда зависит от задачи и прежде чем выбирать это количество необходимо проанализировать входные данные.
