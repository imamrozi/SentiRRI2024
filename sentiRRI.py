import pandas as pd
import numpy as np
import string
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold

try:
    data = pd.read_csv('DATASET-LABELLED-06-06-2024.csv', encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv('DATASET-LABELLED-06-06-2024.csv', encoding='latin-1')
data.head()

data.sentimen.value_counts()
data.Layanan.value_counts()
data.Program.value_counts()
data.Teknis.value_counts()

#PREPROCESSING
#hapus tanda baca dengan mempertahankan spasi
def remove_punctuation_with_space(text):
    return ''.join(ch if ch not in string.punctuation else ' ' for ch in text)
#hapus karakter khusus dalam tweet dengan mempertahankan spasi
def remove_tweet_special(text):
    text = str(text).replace('\\t', "").replace('\\u', "").replace('\\', "")
    text = str(text).encode('ascii', 'replace').decode('ascii')
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", str(text)).split())
    return str(text).replace("http://", " ").replace("https://", " ")
#hapus angka
def remove_number(text):
    return re.sub(r"\d+", "", text)
#hapus spasi ganda
def remove_whitespace_multiple_with_space(text):
    return re.sub('\s+', ' ', text)
#hapus karakter tunggal
def remove_single_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)
#hapus tanda tanya
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
#case folding dan preprocessing awal pada kolom 'text_clean'
data['preprocessed'] = data['Text'].str.lower()
data['preprocessed'] = data['preprocessed'].apply(remove_punctuation_with_space)
data['preprocessed'] = data['preprocessed'].apply(remove_tweet_special)
data['preprocessed'] = data['preprocessed'].apply(remove_number)
data['preprocessed'] = data['preprocessed'].apply(remove_whitespace_multiple_with_space)
data['preprocessed'] = data['preprocessed'].apply(remove_single_char)
data['preprocessed'] = data['preprocessed'].apply(remove_question_marks)
# Mengaplikasikan fungsi 'remove' ke kolom 'Tweet'
data['Text'] = data['Text'].apply(lambda x: remove(x))
# Mengurutkan dan menghapus duplikat berdasarkan kolom 'Tweet'
data.sort_values("Text", inplace=True)
data.drop_duplicates(subset="Text", keep='first', inplace=True)
# Menampilkan hasil pemrosesan
data.head()

#TOKENIZING
# tokenizing
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
def word_tokenize_wrapper(text):
  return word_tokenize(text)
data['tweet_tokens'] = data['preprocessed'].apply(word_tokenize_wrapper)
print('Tokenizing Result:\n')
print(data['tweet_tokens'].head())

# Stopword removal / filtering
nltk.download('stopwords')
from nltk.corpus import stopwords
#Indonesian stopwords from NLTK corpus
stopwords_indonesian = set(stopwords.words('indonesian'))
#additional stopwords
additional_stopwords = ["yg", "dg", "rt", "dgn", "ny", "d", "klo", "kalo", "amp", "biar", "bikin", "bilang", "gak", "krn", "nya", "nih", "sih", "ga", "si", "tau", "tdk", "tuh", "utk", "ya", 'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt', '&amp', 'yah']
stopwords_indonesian.update(additional_stopwords)
# Function for stopwords removal
def stopwords_removal(words):
    words = words.split()
    return [word for word in words if word not in stopwords_indonesian]
# Apply stopwords removal to the 'tweet_tokens' column
data['tweet_nostopwords'] = data['preprocessed'].apply(stopwords_removal)
print(data['tweet_nostopwords'].head())
data.head()

# Stemming
# Inisialisasi Stemmer
stem_factory = StemmerFactory()
Stemmer = stem_factory.create_stemmer()
# Fungsi untuk melakukan stemming pada daftar kata-kata (tokens)
def stemming_text(tokens):
    hasil = [Stemmer.stem(token) for token in tokens]
    return hasil
# Menggunakan fungsi stemming_text pada kolom 'tweet_StopWord' dan menyimpan hasilnya dalam kolom 'stemmed_tokens'
data['stemmed'] = data['tweet_nostopwords'].apply(stemming_text)
# Menampilkan hasil
data.head()

def to_string(tokens):
    hasil = ' '.join(tokens)
    return hasil
# Menggunakan fungsi stemming_text pada kolom 'tweet_StopWord' dan menyimpan hasilnya dalam kolom 'stemmed_tokens'
# data['tweet_Stopword_string'] = data['tweet_StopWord'].apply(to_string)
data['stemmed_string'] = data['stemmed'].apply(to_string)
# Menampilkan hasil
data.head(10)

save = data.to_csv('dataclean.csv', index=False)

# Coba membaca data dengan encoding 'utf-8'
try:
    data = pd.read_csv('dataclean.csv', encoding='utf-8')
except UnicodeDecodeError:
    # Jika terjadi UnicodeDecodeError, coba dengan encoding 'latin-1'
    data = pd.read_csv('dataclean.csv', encoding='latin-1')
# Sekarang data sudah terbaca, dan Anda dapat melanjutkan pemrosesan data
data.head(10)

#TF-IDF
tfidf_vect = TfidfVectorizer()
tfidf_vect.fit_transform(data['stemmed'])
# get idf values
#print('\nidf values:')
#for ele1, ele2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
#	print(ele1, ':', ele2)
all_X_tfidf = tfidf_vect.transform(data['stemmed'])
all_Y = data['sentimen']
#split data
x_train, x_test, y_train, y_test = train_test_split(data['stemmed'], data['sentimen'], test_size=0.1, random_state=42)
x_train_tfidf = tfidf_vect.transform(x_train)
x_test_tfidf = tfidf_vect.transform(x_test)
data_latih = len(y_train)
data_test = len(y_test)
all_data = data_latih+data_test
print('Total Keseluruhan Data: ', all_data)
print('Total Data Latih: ', data_latih)
print('Total Data Test: ', data_test)
x_train_tfidf
#print(x_train_tfidf)

#SVM
# Create SVM classifier model
svm_model = SVC(kernel='rbf', C=1.0, gamma=1.0)
svm_model.fit(x_train_tfidf, y_train)

# Define parameter grid
parameter_grid = {
    'C': np.arange(1, 10, 1),
    'gamma': np.arange(1, 10, 1),
    'kernel': ['rbf', 'linear']
}
# Setup Grid Search CV
svc = SVC()
grid_search = GridSearchCV(estimator=svc, param_grid=parameter_grid, cv=5)
# Fit model
grid_search.fit(x_train_tfidf, y_train)
# Get best parameters and score
cv_results = grid_search.cv_results_
# Finding the best index
best_index = np.argmax(cv_results['mean_test_score'])
best_params = cv_results['params'][best_index]
best_score = cv_results['mean_test_score'][best_index]
print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Score: {best_score}")

# Melatih ulang model dengan best_params
best_model = SVC(**best_params)
best_model.fit(x_train_tfidf, y_train)
test_score = best_model.score(x_test_tfidf, y_test)
print(f"Test Score: {test_score}")

# Memprediksi label pada data uji
y_pred = best_model.predict(x_test_tfidf)
# Menghitung akurasi (Evaluasi Model)
accuracy = accuracy_score(y_test, y_pred)
# Menghitung presisi
precision = precision_score(y_test, y_pred, average='weighted')
# Menghitung recall
recall = recall_score(y_test, y_pred, average='weighted')
# Menghitung f1-score
f1 = f1_score(y_test, y_pred, average='weighted')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

#KFOLD
# Inisialisasi k-fold
k_values = [3, 5, 7, 9, 10]  # Definisikan nilai k yang ingin dievaluasi
accuracies_mean = []
precissions_mean = []
recalls_mean = []
f1_scores_mean = []
svm_model_for_k_fold = SVC(kernel='linear', C=1.0, gamma=1.0)
# Lakukan k-fold cross validation untuk setiap nilai k
for k in k_values:
    kf = KFold(n_splits=k)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    # Lakukan k-fold cross validation
    fold_counter = 1
    for train_index, test_index in kf.split(all_X_tfidf):
        # Bagi data menjadi data latih dan data uji
        X_train, X_test = all_X_tfidf[train_index], all_X_tfidf[test_index]
        y_train, y_test = all_Y.iloc[train_index], all_Y.iloc[test_index]
        # Latih model pada data latih
        svm_model_for_k_fold.fit(X_train, y_train)
        # Prediksi label untuk data uji
        y_pred = svm_model_for_k_fold.predict(X_test)
        # Hitung metrik evaluasi dari prediksi
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        # Simpan metrik evaluasi dalam list
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    # Hitung rata-rata metrik evaluasi dari semua fold
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