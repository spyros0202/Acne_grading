# menu/urls.py
from django.urls import path
from . import views

app = "menu"

urlpatterns = [
    path('', views.menu_choice, name='menu_home'),

]
