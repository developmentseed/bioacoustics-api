from django.urls import path

from .views import sites_list, sites_detail, recordings_detail, recordings_download

urlpatterns = [
    path('sites/', sites_list, name='sites-list'),
    path('sites/<int:id>', sites_detail, name='sites-detail'),
    path('audio_recordings/<int:id>', recordings_detail, name='recordings-detail'),
    path(
        'audio_recordings/download/<str:extension>/<int:id>',
        recordings_download,
        name='recordings-download'
    ),
]
