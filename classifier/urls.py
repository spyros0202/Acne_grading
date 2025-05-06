from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.classify_image, name='classify_image'),
]
