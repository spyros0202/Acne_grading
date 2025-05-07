from django.urls import path
from .views import homepage, register, user_login, user_logout

app_name = 'users'

urlpatterns = [
    path('', homepage, name="home"),  # Root path
    path('register/', register, name="register"),
    path('login/', user_login, name="login"),
    path('user-logout/', user_logout, name="logout"),
]
