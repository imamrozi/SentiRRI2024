from django.urls import path
from . import views

app_name = "sentimen"
urlpatterns = [
    path("dashboad/", views.dashboard, name="dashboard"),
    path("dataset/", views.dataset, name="dataset"),
    # path('preprocessing/', views.preprocessing, name="preprocessing"),
    # path("preprocessing1/<str:proses>/", views.preprocessing1, name="preprocessing1"),
    # path("preprocessing1/", views.preprocessing1, name="preprocessing2"),

    # path("preprocessing/<int:proces>/", views.preprocessing, name="preprocessing"),
    path("preprocessing/", views.preprocessing, name="preprocessing"),
    # path("training/", views.training, name="training"),


    path("preprocessing/stemming/", views.stemming, name="stemming"),
    path("preprocessing/stopword/", views.stopwords, name="stopword"),
    path("preprocessing/stemmed_stopword/", views.stemmed_stopwords, name="stemmed_stopword"),

    path("pengujian/", views.pengujian, name="pengujian"),
    path("prediksi/", views.prediksi, name="prediksi"),
    path("visualisasi/", views.visualisasi, name="visualisasi"),
]