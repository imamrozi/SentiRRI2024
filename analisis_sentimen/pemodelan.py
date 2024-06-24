from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from joblib import dump
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# PEMBOBOTAN 
# TF -IDF 
def tf_idf(preprocessing, sentimen):
    tfidf_vect = TfidfVectorizer()
    tfidf_vect.fit_transform(preprocessing)
    all_text_vectorized = tfidf_vect.transform(preprocessing)
    all_label = sentimen 
    return all_text_vectorized, all_label

# SPLIT DATA 
def split(all_text_vectorized, all_label):
    train_text_vectorized, test_text_vectorized, train_label, test_label = train_test_split(all_text_vectorized, all_label, test_size=0.1, random_state=42)
    data_latih = len(train_label)
    data_test = len(test_label)
    all_data = data_latih+data_test

    return train_text_vectorized, test_text_vectorized, train_label, test_label, data_latih, data_test, all_data

# PEMODELAN dengan SVM 
def model_svm(preprocessing, sentimen):
    # all_text_vectorized, all_label = tf_idf(preprocessing, sentimen)
    train_text_vectorized, test_text_vectorized, train_label, test_label, data_latih, data_test, all_data = split(tf_idf(preprocessing, sentimen))

    best_model = SVC(kernel='rbf', C=10.0, gamma=0.1)
    best_model.fit(train_text_vectorized, train_label)
    test_score = best_model.score(test_text_vectorized, test_label)

    # simpan model
    dump(best_model, 'analisis_sentimen/model_ml/svm_joblib.joblib')
    return test_score

