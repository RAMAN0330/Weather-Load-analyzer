from django.urls import path, include

urlpatterns = [
    path("", include("pipeline.urls")),
    path("auth/", include("accounts.urls")),
]
