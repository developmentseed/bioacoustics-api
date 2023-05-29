from django.urls import path

from .views import sites_list, sites_detail, recordings_detail

urlpatterns = [
    path('sites/', sites_list, name='sites-list'),
    path('sites/<int:id>', sites_detail, name='sites-detail'),
    path('audio_recordings/<int:id>', recordings_detail, name='recordings-detail'),
]