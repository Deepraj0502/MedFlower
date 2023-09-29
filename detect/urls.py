from django.urls import include, path
from . import views

urlpatterns = [
    path("", views.home),
    path("info", views.info),
    path("mediuse", views.mediuse),
]
