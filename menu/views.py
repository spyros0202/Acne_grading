# menu/views.py
from django.shortcuts import render

import os
from django.conf import settings
from django.shortcuts import render

def menu_choice(request):
    return render(request, 'menu.html')