from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('predict/', views.predict, name='predict'),
    path('predict/api/', views.predict_api, name='predict_api'),
]
