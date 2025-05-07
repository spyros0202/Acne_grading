from django.shortcuts import render,  redirect
from . forms import UserForm, LoginForm
from django.contrib.auth import authenticate, login ,logout

def homepage(request):

    return render(request, 'users/home.html', {})


# users/views.py

def user_login(request):
    form = LoginForm()

    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)

        if form.is_valid():
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password=password)

            if user is not None:
                login(request, user)

                # Redirect to the next URL or default to the BMI page
                next_url = request.GET.get('next', 'menu:menu_home')
                return redirect(next_url)

    context = {'loginform': form}
    return render(request, 'users/login.html', context=context)


def register(request):

    form = UserForm()

    if request.method == "POST":
        form = UserForm(request.POST)

        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(request, username=username, password=raw_password)
            if user is not None:
                login(request, user)
                return redirect("menu:menu_home")


    context={'registerform': form}

    return render(request, 'users/register.html', context=context)



def user_logout(request):
    logout(request)
    return redirect('users:home')
