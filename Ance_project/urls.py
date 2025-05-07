from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('users.urls')),
    path('menu/', include('menu.urls')),
    path('classifier/', include('classifier.urls')),
]
