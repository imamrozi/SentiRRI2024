#IMPORT 
import pandas as pd
import numpy as np
import string
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
from gensim.models import Word2Vec
from gensim.models import FastText
#END IMPORT 

#READ DATA
try:
    data = pd.read_csv('DATASET-LABELLED-06-06-2024.csv', encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv('DATASET-LABELLED-06-06-2024.csv', encoding='latin-1')
data.head()
#data.sentimen.value_counts()
#data.Layanan.value_counts()
#data.Program.value_counts()
#data.Teknis.value_counts()
#END READ DATA

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

#PREPROCESSING
data['preprocessed'] = data['Text'].str.lower()
data['preprocessed'] = data['preprocessed'].apply(remove_punctuation_with_space)
data['preprocessed'] = data['preprocessed'].apply(remove_tweet_special)
data['preprocessed'] = data['preprocessed'].apply(remove_number)
data['preprocessed'] = data['preprocessed'].apply(remove_whitespace_multiple_with_space)
data['preprocessed'] = data['preprocessed'].apply(remove_single_char)
data['preprocessed'] = data['preprocessed'].apply(remove_question_marks)
#apply 'remove' to 'Text'
data['Text'] = data['Text'].apply(lambda x: remove(x))
data.sort_values("Text", inplace=True)
data.drop_duplicates(subset="Text", keep='first', inplace=True)
data.head()
#END PREPROCESSING

#TOKENIZATION
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
def word_tokenize_wrapper(text):
  return word_tokenize(text)
data['tweet_tokens'] = data['preprocessed'].apply(word_tokenize_wrapper)
print('Tokenizing Result:\n')
print(data['tweet_tokens'].head())
#END TOKENIZATION

#STEMMING WITHOUT REMOVING STOPWORD
stem_factory = StemmerFactory()
Stemmer = stem_factory.create_stemmer()
def stemming_text(tokens):
    hasil = [Stemmer.stem(token) for token in tokens]
    return hasil
data['stemmed_only'] = data['tweet_tokens'].apply(stemming_text)
data.head()
#END STEMMING

#REMOVING STOPWORD WITHOUT STEMMING
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_indonesian = set(stopwords.words('indonesian'))
additional_stopwords = ["yg", "dg", "rt", "dgn", "ny", "d", "klo", "kalo", "amp", "biar", "bikin", "bilang", "gak", "krn", "nya", "nih", "sih", "ga", "si", "tau", "tdk", "tuh", "utk", "ya", 'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt', '&amp', 'yah']
stopwords_indonesian.update(additional_stopwords)
def stopwords_removal(words):
    words = words.split()
    return [word for word in words if word not in stopwords_indonesian]
data['stopword_only'] = data['preprocessed'].apply(stopwords_removal)
print(data['stopword_only'].head())
data.head()
#REMOVING STOPWORD AND STEMMING

#STEMMING AND STOPWORD REMOVAL
stem_factory = StemmerFactory()
Stemmer = stem_factory.create_stemmer()
def stemming_text(tokens):
    hasil = [Stemmer.stem(token) for token in tokens]
    return hasil
data['stemmed_stopword'] = data['stopword_only'].apply(stemming_text)
data.head()
#END STEMMING

def to_string(tokens):
    hasil = ' '.join(tokens)
    return hasil
# Menggunakan fungsi stemming_text pada kolom 'tweet_StopWord' dan menyimpan hasilnya dalam kolom 'stemmed_tokens'
# data['tweet_Stopword_string'] = data['tweet_StopWord'].apply(to_string)
data['stemmed_only_string'] = data['stemmed_only'].apply(to_string)
data['stopword_only_string'] = data['stopword_only'].apply(to_string)
data['stemmed_stopword_string'] = data['stemmed_stopword'].apply(to_string)
# Menampilkan hasil
data.head(10)

save = data.to_csv('dataclean_new.csv', index=False)

#READ CLEAN DATA 
try:
    data = pd.read_csv('dataclean_new.csv', encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv('dataclean_new.csv', encoding='latin-1')
data.head(10)
#END READ CLEAN DATA 

#CREATE TF-IDF MODEL
tfidf_vect = TfidfVectorizer()
tfidf_vect.fit_transform(data['stemmed'])
all_text_vectorized = tfidf_vect.transform(data['stemmed'])
all_label = data['sentimen']
#END CREATE TF-IDF MODEL

#W2V and FASTTEXT MODEL
#w2v_model = Word2Vec.load('idwiki_word2vec_400_new_lower.model');
#import fasttext
#w2v_model = FastText.load_fasttext_format('cc.id.300.bin')
#w2v_model = FastText.load_fasttext_format('cc.id.300.bin')
# def text_to_vector(tokens):
#     if isinstance(tokens, str):
#       words = tokens.split()
#       vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
#       if len(vectors) > 0:
#         return np.mean(vectors, axis=0)
#       else:
#         return np.zeros(w2v_model.vector_size)
#     else:
#       return np.zeros(w2v_model.vector_size)
# all_text_vectorized = np.array([text_to_vector(text) for text in data['tweet_tokens']])
# all_label = data['sentimen']
#END W2V MODEL

#SPLIT DATA
train_text_vectorized, test_text_vectorized, train_label, test_label = train_test_split(all_text_vectorized, all_label, test_size=0.1, random_state=42)
data_latih = len(train_label)
data_test = len(test_label)
all_data = data_latih+data_test
print('Total Keseluruhan Data: ', all_data)
print('Total Data Latih: ', data_latih)
print('Total Data Test: ', data_test)
#print(x_train_tfidf)

#SVM MODEL
#svm_model = SVC(kernel='rbf', C=1.0, gamma=1.0)
#svm_model.fit(train_text_vectorized, train_label)
#END SVM MODEL

#GRID SEARCH TO FIND BEST PARAM
# parameter_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
#     'degree': [2, 3, 4, 5]  # Hanya digunakan untuk 'poly' kernel
# }
# svc = SVC()
# grid_search = GridSearchCV(estimator=svc, param_grid=parameter_grid, cv=10)
# grid_search.fit(all_text_vectorized, all_label)
# cv_results = grid_search.cv_results_
# best_index = np.argmax(cv_results['mean_test_score'])
# best_params = cv_results['params'][best_index]
# best_score = cv_results['mean_test_score'][best_index]
# print(f"Best Parameters: {best_params}")
# print(f"Best Cross-Validation Score: {best_score}")
#GRID SEARCH TO FIND BEST PARAM

#RANDOM SEARCH FOR HYPERPARAM TUNING
# param = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
#     'degree': [2, 3, 4, 5]  # Hanya digunakan untuk 'poly' kernel
# }
# svc = SVC()
# random_search = RandomizedSearchCV(estimator=svc,param_distributions=param,cv=10)
# random_search.fit(all_text_vectorized, all_label)
# cv_results = random_search.cv_results_
# best_index = np.argmax(cv_results['mean_test_score'])
# best_params = cv_results['params'][best_index]
# best_score = cv_results['mean_test_score'][best_index]
# print(f"Best Parameters: {best_params}")
# print(f"Best Cross-Validation Score: {best_score}")
#

#TRAIN BEST MODEL
#best_model = SVC(**best_params)
best_model = SVC(kernel='rbf', C=10.0, gamma=0.1)
best_model.fit(train_text_vectorized, train_label)
test_score = best_model.score(test_text_vectorized, test_label)
print(f"Test Score: {test_score}")
#END TRAIN BEST MODEL

#PREDICT 
pred_label = best_model.predict(test_text_vectorized)
accuracy = accuracy_score(test_label, pred_label)
precision = precision_score(test_label, pred_label, average='weighted')
recall = recall_score(test_label, pred_label, average='weighted')
f1 = f1_score(test_label, pred_label, average='weighted')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
#END PREDICT
 

#K-FOLD
# k_values = [3, 5, 7, 9, 10]
k_values = [10]
accuracies_mean = []
precissions_mean = []
recalls_mean = []
f1_scores_mean = []
#svm_model_for_k_fold = SVC(kernel='linear', C=1.0, gamma=1.0)
for k in k_values:
    kf = KFold(n_splits=k)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    fold_counter = 1
    for train_index, test_index in kf.split(all_text_vectorized):
        train_text_vectorized, test_text_vectorized = all_text_vectorized[train_index], all_text_vectorized[test_index]
        train_label, test_label = all_label.iloc[train_index], all_label.iloc[test_index]
        best_model.fit(train_text_vectorized, train_label)
        pred_label = best_model.predict(test_text_vectorized)
        accuracy = accuracy_score(test_label, pred_label)
        precision = precision_score(test_label, pred_label, average='weighted',zero_division=0)
        recall = recall_score(test_label, pred_label, average='weighted',zero_division=0)
        f1 = f1_score(test_label, pred_label, average='weighted',zero_division=0)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    average_accuracy = sum(accuracies) / len(accuracies)
    average_precision = sum(precisions) / len(precisions)
    average_recall = sum(recalls) / len(recalls)
    average_f1_score = sum(f1_scores) / len(f1_scores)
    # Gabungkan rata2 akurasi, presisi, recall dan f1score dari tiap k-fold
    accuracies_mean.append(average_accuracy)
    precissions_mean.append(average_precision)
    recalls_mean.append(average_recall)
    f1_scores_mean.append(average_f1_score)
    # Cetak rata-rata metrik evaluasi
    print("k-fold dengan k = ", k)
    print("Rata-rata akurasi k-fold:", average_accuracy)
    print("Rata-rata precision k-fold:", average_precision)
    print("Rata-rata recall k-fold:", average_recall)
    print("Rata-rata F1 Score k-fold:", average_f1_score)
# Hitung rata2 dari semua K-Fold
average_all_accuracy = sum(accuracies_mean) / len(accuracies_mean)
average_all_precision = sum(precissions_mean) / len(precissions_mean)
average_all_recall = sum(recalls_mean) / len(recalls_mean)
average_all_f1_score = sum(f1_scores_mean) / len(f1_scores_mean)
# Cetak rata-rata metrik evaluasi
print("AKUMULASI DARI SEMUA K-FOLD")
print("Rata-rata akurasi k-fold:", average_all_accuracy)
print("Rata-rata precision k-fold:", average_all_precision)
print("Rata-rata recall k-fold:", average_all_recall)
print("Rata-rata F1 Score k-fold:", average_all_f1_score)
#END K-FOLD