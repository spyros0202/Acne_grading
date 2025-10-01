# menu/views.py
from django.shortcuts import render
# from django.contrib.auth.decorators import login_required

import os
from django.conf import settings
from django.shortcuts import render

# @login_required

def menu_choice(request):
    return render(request, 'menu/menu.html')