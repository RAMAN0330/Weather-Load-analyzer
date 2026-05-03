from django.urls import include, path

urlpatterns = [
    path("", include("pipeline.urls")),
    path("auth/", include("accounts.urls")),
]
