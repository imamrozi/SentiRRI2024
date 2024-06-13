from multiprocessing import context
from django.shortcuts import render 

def index(request):
    context = {
        'title':'Login',
    }
    return render(request, 'login.html', context)