from django.urls import path
from . import views

app_name = "sentimen"
urlpatterns = [
    path('dashboad/', views.dashboard, name="dashboard"),
    path('dataset/', views.dataset, name="dataset"),
    path('preprocessing/', views.preprocessing, name="preprocessing"),
    path('pengujian/', views.pegujian, name="pengujian"),
    path('visualisasi/', views.visualisasi, name="visualisasi"),
]