from django.urls import path

from . import views

urlpatterns = [
    path("register",       views.RegisterView.as_view(),    name="auth-register"),
    path("login",          views.LoginView.as_view(),       name="auth-login"),
    path("logout",         views.LogoutView.as_view(),      name="auth-logout"),
    path("me",             views.MeView.as_view(),          name="auth-me"),
    path("results/save",   views.SaveResultView.as_view(),  name="auth-save-result"),
    path("results",        views.ListResultsView.as_view(), name="auth-list-results"),
    path("weights/save",   views.SaveWeightsView.as_view(), name="auth-save-weights"),
    path("weights",        views.ListWeightsView.as_view(), name="auth-list-weights"),
]
