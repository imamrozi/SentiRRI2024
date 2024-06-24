from django.shortcuts import redirect
import pandas as pd
import numpy as np
import string
import re
import nltk 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

#DEFINE PREPROCESSING METHOD
#remove punctuation except space
def remove_punctuation_with_space(text):
    return ''.join(ch if ch not in string.punctuation else ' ' for ch in text)

#remove special char except space
def remove_tweet_special(text):
    text = str(text).replace('\\t', "").replace('\\u', "").replace('\\', "")
    text = str(text).encode('ascii', 'replace').decode('ascii')
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", str(text)).split())
    return str(text).replace("http://", " ").replace("https://", " ")

#remove number
def remove_number(text):
    return re.sub(r"\d+", "", text)

#remove double space
def remove_whitespace_multiple_with_space(text):
    return re.sub('\s+', ' ', text)

#remove single char
def remove_single_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

#remove ?
def remove_question_marks(text):
    return text.replace("?", "")

#bersihkan teks tambahan pada kolom 'Tweet'
def remove(tweet):
    # Remove angka
    tweet = re.sub('[0-9]+', '', tweet)
    # Remove stock market tickers like $GE
    tweet = re.sub(r'^\$\w*', '', tweet)
    # Remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # Remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # Remove hashtags
    tweet = re.sub(r'#', '', tweet)
    # Remove commas
    tweet = re.sub(r',', '', tweet)
    # Remove specific substrings (x, xf, xe, xa)
    tweet = re.sub(r'\b(xf|xe|xa|x)\b', '', tweet)
    return tweet
#END DEFINE PREPROCESSING METHOD

#TOKENIZATION
def word_tokenize_wrapper(text):
  return word_tokenize(text)

#STEMMING WITHOUT REMOVING STOPWORD
stem_factory = StemmerFactory()
Stemmer = stem_factory.create_stemmer()
def stemming_text(tokens):
    hasil = [Stemmer.stem(token) for token in tokens]
    return hasil

#REMOVING STOPWORD WITHOUT STEMMING
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_indonesian = set(stopwords.words('indonesian'))
additional_stopwords = ["yg", "dg", "rt", "dgn", "ny", "d", "klo", "kalo", "amp", "biar", "bikin", "bilang", "gak", "krn", "nya", "nih", "sih", "ga", "si", "tau", "tdk", "tuh", "utk", "ya", 'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt', '&amp', 'yah']
stopwords_indonesian.update(additional_stopwords)
def stopwords_removal(words):
    words = words.split()
    return [word for word in words if word not in stopwords_indonesian]


#REMOVING STOPWORD AND STEMMING
#STEMMING AND STOPWORD REMOVAL
stem_factory = StemmerFactory()
Stemmer = stem_factory.create_stemmer()
def stemming_text(tokens):
    hasil = [Stemmer.stem(token) for token in tokens]
    return hasil

# To String 
def to_string(tokens):
    hasil = ' '.join(tokens)
    return hasil