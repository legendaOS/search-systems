
FILE_NAME = 'C:\GitGub reps\search-systems\lab1\lab1\labeled.csv'

import pandas as pd

# Read CSV file into DataFrame
train = pd.read_csv(FILE_NAME)


print(train.head())

# разбить на слова

import nltk
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('russian')

from nltk import word_tokenize
nltk.download('punkt')

words_in_texts = []

from nltk.corpus import stopwords
stops = stopwords.words('russian')

# чистим текст от стоп слов и записываем их леммы
real_stroki = []

import re
def lemmatize(doc):
    patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
    doc = re.sub(patterns, ' ', doc)
    return doc

for stroka in train.comment:
    
    buf_words = []
    
    words = word_tokenize(stroka)

    for one_word in words:
        if not one_word in stops:
            if lemmatize(one_word) != ' ':
                buf_words.append(lemmatize(one_word))

    real_stroki.append(buf_words)



# готово real_stroki
print(real_stroki[0])


#преобразовать в список строк
list_of_stroki = [ ' '.join(i) for i in real_stroki ]

print(list_of_stroki[0])



#векторизация

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(list_of_stroki)


vectorizer.fit(list_of_stroki)
# summarize
# print(vectorizer.vocabulary_)
# encode document

vector = vectorizer.transform(list_of_stroki)
# summarize encoded vector
print(vector.shape)
print(type(vector))
# print(vector.toarray())


# создание выборок и обучение модели

X = vector.toarray()
y = train.toxic

# обучение модели

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)


# проба модели

input_string = 'котики такие милые'
lol = vectorizer.transform([input_string])

print('------------------------')
print(lol)
print(lol.toarray())

print('------------------------')
print(neigh.predict(lol.toarray()))






