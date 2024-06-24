from concurrent.futures import process
from multiprocessing import context
from pyexpat.errors import messages
from django.shortcuts import render, redirect
import pandas as pd
import io
import csv


from .forms import UploadFileForm
from .preprocessing import remove, remove_number, remove_single_char, remove_tweet_special
from .preprocessing import remove_question_marks, remove_punctuation_with_space, remove_whitespace_multiple_with_space
from .preprocessing import word_tokenize_wrapper, stemming_text ,stopwords_removal, to_string
# from .pemodelan import tf_idf

# View Dashboard
def dashboard(request):
    context = {
        'title':'Dashboard Sentimen Analisis',
    }
    return render(request, 'dashboard.html', context)

# Cleaning Data
# Preprocessing
def clean_data(request):
    return render(request, 'dataset.html')

# View Dataset
def dataset(request):
    dataset = []
    post_form = UploadFileForm(request.POST, request.FILES)
    if request.method == "POST":
        if post_form.is_valid():
            data = request.FILES["data_csv"]
            decode_file = data.read().decode('latin-1')
            io_string = io.StringIO(decode_file)
            reader = csv.reader(io_string, delimiter=',')
            for row in reader:
                dataset.append(row)

            # simpan data ke session
            request.session['dataset'] = dataset
            # return redirect('sentimen:preprocessing')
        else:
            messages.error(request, 'Data Tidak Sesuai')
    else:
        post_form = UploadFileForm(request.POST, request.FILES)

    context = {
        'title':'Dataset Sentimen Analisis',
        'data_form':post_form,
        'dataset': dataset

    }
    return render(request, 'dataset.html', context)


def dataset1(request):
    post_form = UploadFileForm(request.POST, request.FILES)
    if request.method == "POST":
        if post_form.is_valid():
            # print(post_form(request.FILES["data_csv"]))
            data_lower = (request.FILES["data_csv"]).str.lower()
            data_remove_punctuation = remove_punctuation_with_space(data_lower)
            data_remove_tweet_special = remove_tweet_special(data_remove_punctuation)
            data_remove_number = remove_number(data_remove_tweet_special)
            data_remove_whitespace = remove_whitespace_multiple_with_space(data_remove_number)
            data_remove_single_char = remove_single_char(data_remove_whitespace)
            data_remove_question_marks = remove_question_marks(data_remove_single_char)
            data_remove_text = remove(data_remove_question_marks)
            
            context = {
                'title':'Dataset Sentimen Analisis',
                'hasil_preprocessing':data_remove_text,
            }
            return render(request, 'dataset.html', context)
        else:
            messages.error(request, 'Data Tidak Sesuai')
    else:
        post_form = UploadFileForm(request.POST, request.FILES)
        
    context = {
        'title':'Dataset Sentimen Analisis',
        'data_form':post_form,
    }
    return render(request, 'dataset.html', context)

# View Preprocessing data 
# def preprocessing(request):
#     dataset = request.session.get('dataset', [])

#     dataset_hasil = [row for row in dataset]

#     context = {
#         'title':'Preprocessing Data',
#         'dataset':dataset_hasil
#     }
#     return render(request, 'preprocecing.html', context)

# View function
def preprocessing1(request, proces=None):
    dataset = request.session.get('dataset', [])

    if not dataset:
        return render(request, 'preprocecing.html', {'title': 'Preprocessing Data', 'dataset': []})

    df = pd.DataFrame(dataset[1:], columns=dataset[0])

    # Preprocessing
    df['preprocessed'] = df['Text'].str.lower()
    df['preprocessed'] = df['preprocessed'].apply(remove_punctuation_with_space)
    df['preprocessed'] = df['preprocessed'].apply(remove_tweet_special)
    df['preprocessed'] = df['preprocessed'].apply(remove_number)
    df['preprocessed'] = df['preprocessed'].apply(remove_whitespace_multiple_with_space)
    df['preprocessed'] = df['preprocessed'].apply(remove_single_char)
    df['preprocessed'] = df['preprocessed'].apply(remove_question_marks)

    # Apply 'remove' to 'Text'
    df['Text'] = df['Text'].apply(remove)
    df.sort_values("Text", inplace=True)
    df.drop_duplicates(subset="Text", keep="first", inplace=True)

    # Tokenization
    df['tweet_tokens'] = df['preprocessed'].apply(word_tokenize_wrapper)

    if proces == "1":
        # Stemming 
        df['stemmed_only'] = df['tweet_tokens'].apply(stemming_text)
        # To String
        df['stemmed_only_string'] = df['stemmed_only'].apply(to_string)

    elif proces == "2":
        # Stopword 
        df['stopword_only'] = df['preprocessed'].apply(stopwords_removal)
        # To String 
        df['stopword_only_string'] = df['stopword_only'].apply(to_string)

    elif proces == "3":
        # Stopword 
        df['stopword_only'] = df['preprocessed'].apply(stopwords_removal)
        # Stemming 
        df['stemmed_stopword'] = df['stopword_only'].apply(stemming_text)
        # To string 
        df['stemmed_stopword_string'] = df['stemmed_stopword'].apply(to_string)

    dataset_hasil = df.values.tolist()

    context = {
        'title': 'Preprocessing Data',
        'dataset': dataset_hasil
    }
    return render(request, 'preprocecing.html', context)

# Preprocessing 
def preprocessing(request):
    dataset = request.session.get('dataset', [])
    total_data = 0
    headers = []
    dataset_hasil = []
    proses_pilih= ""
    selected_value = ""

    if dataset:

        df = pd.DataFrame(dataset[1:], columns=dataset[0])

        #PREPROCESSING
        df['preprocessed'] = df['Text'].str.lower()
        df['preprocessed'] = df['preprocessed'].apply(remove_punctuation_with_space)
        df['preprocessed'] = df['preprocessed'].apply(remove_tweet_special)
        df['preprocessed'] = df['preprocessed'].apply(remove_number)
        df['preprocessed'] = df['preprocessed'].apply(remove_whitespace_multiple_with_space)
        df['preprocessed'] = df['preprocessed'].apply(remove_single_char)
        df['preprocessed'] = df['preprocessed'].apply(remove_question_marks)

        #apply 'remove' to 'Text'
        df['Text'] = df['Text'].apply(lambda x: remove(x))
        df.sort_values("Text", inplace=True)
        df.drop_duplicates(subset="Text", keep="first", inplace=True)
        # df = df.values.tolist()
        # print(df[0])

        # TOKENIZATION ???
        df['tweet_tokens'] = df['preprocessed'].apply(word_tokenize_wrapper)
        print(df.head())

        if request.method == "POST":
            selected_value = request.POST.get('value', "")
            proces = int(selected_value)
            # proces = int(request.POST.get('value',0))

            if proces == 1:
                proses_pilih = "Stemming"
                # Stemming 
                df['stemmed_only'] = df['tweet_tokens'].apply(stemming_text)
                print(df.head())

                # To String
                df['stemmed_only_string'] = df['stemmed_only'].apply(to_string)
                print(df.head())

            elif proces == 2:
                proses_pilih = "Stopwords"
                # Stopword 
                df['stopword_only'] = df['preprocessed'].apply(stopwords_removal)
                print(df.head())

                # to String 
                df['stopword_only_string'] = df['stopword_only'].apply(to_string)
                print(df.head())

            elif proces == 3:
                proses_pilih = "Stemming Stopwords"
                # Stopword 
                df['stopword_only'] = df['preprocessed'].apply(stopwords_removal)
                print(df.head())

                # Stemming 
                df['stemmed_stopword'] = df['stopword_only'].apply(stemming_text)
                print(df.head())

                # to string 
                df['stemmed_stopword_string'] = df['stemmed_stopword'].apply(to_string)
                print(df.head())
        
        total_data = len(df)
        headers = df.columns.tolist()
        print(headers)
        dataset_hasil = df.values.tolist()
        print(dataset_hasil[0])

        # simpan data ke session
        dataset_preprocessing = df.to_dict(orient='records')
        request.session['preprocessing'] = dataset_preprocessing
        # return redirect('sentimen:pengujian')

    context = {
        'title':'Preprocessing Data',
        'total':total_data,
        'preproses':proses_pilih,
        'selected_value':selected_value,
        'dataset':dataset_hasil,
        'header': headers,
    }
    return render(request, 'preprocecing.html', context)
 

# Stemming AJAH !!!
def stemming(request):
    dataset = request.session.get('dataset', [])

    # dataset_hasil = [row for row in dataset]

    if dataset:
        df = pd.DataFrame(dataset[1:], columns=dataset[0])

        #PREPROCESSING
        df['preprocessed'] = df['Text'].str.lower()
        df['preprocessed'] = df['preprocessed'].apply(remove_punctuation_with_space)
        df['preprocessed'] = df['preprocessed'].apply(remove_tweet_special)
        df['preprocessed'] = df['preprocessed'].apply(remove_number)
        df['preprocessed'] = df['preprocessed'].apply(remove_whitespace_multiple_with_space)
        df['preprocessed'] = df['preprocessed'].apply(remove_single_char)
        df['preprocessed'] = df['preprocessed'].apply(remove_question_marks)

        #apply 'remove' to 'Text'
        df['Text'] = df['Text'].apply(lambda x: remove(x))
        df.sort_values("Text", inplace=True)
        df.drop_duplicates(subset="Text", keep="first", inplace=True)
        # df = df.values.tolist()
        # print(df[0])

        # TOKENIZATION
        df['tweet_tokens'] = df['preprocessed'].apply(word_tokenize_wrapper)
        print(df.head())
        
        # Stemming 
        df['stemmed_only'] = df['tweet_tokens'].apply(stemming_text)
        print(df.head())

        # To String
        df['stemmed_only_string'] = df['stemmed_only'].apply(to_string)
        print(df.head())

        dataset_hasil = df.values.tolist()
        print(dataset_hasil[0])


    else:
        dataset_hasil = []

    context = {
        'title':'Preprocessing Data',
        'dataset':dataset_hasil
    }
    return render(request, 'preprocecing.html', context)

# Stopwords AJAH !!!
def stopwords(request):
    dataset = request.session.get('dataset', [])

    # dataset_hasil = [row for row in dataset]

    if dataset:
        df = pd.DataFrame(dataset[1:], columns=dataset[0])

        #PREPROCESSING
        df['preprocessed'] = df['Text'].str.lower()
        df['preprocessed'] = df['preprocessed'].apply(remove_punctuation_with_space)
        df['preprocessed'] = df['preprocessed'].apply(remove_tweet_special)
        df['preprocessed'] = df['preprocessed'].apply(remove_number)
        df['preprocessed'] = df['preprocessed'].apply(remove_whitespace_multiple_with_space)
        df['preprocessed'] = df['preprocessed'].apply(remove_single_char)
        df['preprocessed'] = df['preprocessed'].apply(remove_question_marks)

        #apply 'remove' to 'Text'
        df['Text'] = df['Text'].apply(lambda x: remove(x))
        df.sort_values("Text", inplace=True)
        df.drop_duplicates(subset="Text", keep="first", inplace=True)
        # df = df.values.tolist()
        # print(df[0])

        # TOKENIZATION ???
        df['tweet_tokens'] = df['preprocessed'].apply(word_tokenize_wrapper)
        print(df.head())
        
        # Stopwords 
        df['stopword_only'] = df['preprocessed'].apply(stopwords_removal)
        print(df.head())

        # to String 
        df['stopword_only_string'] = df['stopword_only'].apply(to_string)
        print(df.head())

        dataset_hasil = df.values.tolist()
        print(dataset_hasil[0])

    else:
        dataset_hasil = []

    context = {
        'title':'Preprocessing Data',
        'dataset':dataset_hasil
    }
    return render(request, 'preprocecing.html', context)
 
#  STEMMING AND STOPWORD STEMMING 
def stemmed_stopwords(request):
    dataset = request.session.get('dataset', [])

    # dataset_hasil = [row for row in dataset]

    if dataset:
        df = pd.DataFrame(dataset[1:], columns=dataset[0])

        #PREPROCESSING
        df['preprocessed'] = df['Text'].str.lower()
        df['preprocessed'] = df['preprocessed'].apply(remove_punctuation_with_space)
        df['preprocessed'] = df['preprocessed'].apply(remove_tweet_special)
        df['preprocessed'] = df['preprocessed'].apply(remove_number)
        df['preprocessed'] = df['preprocessed'].apply(remove_whitespace_multiple_with_space)
        df['preprocessed'] = df['preprocessed'].apply(remove_single_char)
        df['preprocessed'] = df['preprocessed'].apply(remove_question_marks)

        #apply 'remove' to 'Text'
        df['Text'] = df['Text'].apply(lambda x: remove(x))
        df.sort_values("Text", inplace=True)
        df.drop_duplicates(subset="Text", keep="first", inplace=True)
        # df = df.values.tolist()
        # print(df[0])

        # TOKENIZATION ???
        df['tweet_tokens'] = df['preprocessed'].apply(word_tokenize_wrapper)
        print(df.head())
        
        # Stopword 
        df['stopword_only'] = df['preprocessed'].apply(stopwords_removal)
        print(df.head())

        # Stemming 
        df['stemmed_stopword'] = df['stopword_only'].apply(stemming_text)
        print(df.head())

        # to string 
        df['stemmed_stopword_string'] = df['stemmed_stopword'].apply(to_string)
        print(df.head())

        dataset_hasil = df.values.tolist()
        print(dataset_hasil[0])


    else:
        dataset_hasil = []

    context = {
        'title':'Preprocessing Data',
        'dataset':dataset_hasil
    }
    return render(request, 'preprocecing.html', context)
 
# View Training model 
def pengujian(request):
    headers = []
    dataset_hasil = []

    dataset_preprocessing = request.session.get('preprocessing', [])
    # print(dataset_preprocessing)

    # ubah list ke dalam dataframe
    df = pd.DataFrame(dataset_preprocessing[1:], columns=dataset_preprocessing[0])
    headers = df.columns.tolist()
    dataset_hasil = df.values.tolist()

    print(df.head())

    context = {
        'title':'Preprocessing Data',
        'dataset':dataset_hasil,
        'header':headers
    }
    return render(request, 'pengujian.html', context)


# view Pengujian
def prediksi(request):
    dataset_preprocessing = request.session.get('preprocessing', [])

    df = pd.DataFrame(dataset_preprocessing[1:], columns=dataset_preprocessing[0])

    # PEMBOBOTAN TF-IDF
    # tf_idf_freq = tf_idf(df[:])

    context = {
        'title':'Pengujian Data Sentimen Analisis',
    }
    return render(request, 'pengujian.html', context)

# view visualisasi
def visualisasi(request):
    context = {
        'title':'Visualisasi Data Sentimen Analisis',
    }
    return render(request, 'visualisasi.html', context)

