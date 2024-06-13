from multiprocessing import context
from django.shortcuts import render

# Create your views here.
def dashboard(request):
    context = {
        'title':'Dashboard Sentimen Analisis',
    }
    return render(request, 'dashboard.html', context)

def dataset(request):
    context = {
        'title':'Dataset Sentimen Analisis',
    }
    return render(request, 'dataset.html', context)

def preprocessing(request):
    context = {
        'title':'Preprocessing Data',
    }
    return render(request, 'preprocecing.html', context)

def pegujian(request):
    context = {
        'title':'Pengujian Data Sentimen Analisis',
    }
    return render(request, 'pengujian.html', context)

def visualisasi(request):
    context = {
        'title':'Visualisasi Data Sentimen Analisis',
    }
    return render(request, 'visualisasi.html', context)

