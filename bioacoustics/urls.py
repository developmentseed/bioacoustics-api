from django.conf import settings
from django.urls import path, re_path, include, reverse_lazy
from django.conf.urls.static import static
from django.contrib import admin
from django.views.generic.base import RedirectView
from rest_framework.routers import DefaultRouter
from rest_framework.authtoken import views
from rest_framework.permissions import AllowAny
from rest_framework.schemas import get_schema_view

from .users.views import UserViewSet, UserCreateViewSet
from .milvus.views import search_view, embed_view
from .a2o_api_proxy.urls import urlpatterns

router = DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'users', UserCreateViewSet)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/', include(router.urls)),
    path('api/v1/a2o/', include(urlpatterns)),
    path('api/v1/search/', search_view),
    path('api/v1/embed/', embed_view),
    path('api/v1/openapi/', get_schema_view(
        title="Google Bioacoustics API",
        description="API for Google Bioacoustics project",
        version="1.0.0",
        permission_classes=[AllowAny]
    ), name='openapi-schema'),
    path('api-token-auth/', views.obtain_auth_token),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),

    # the 'api-root' from django rest-frameworks default router
    # http://www.django-rest-framework.org/api-guide/routers/#defaultrouter
    re_path(r'^$', RedirectView.as_view(url=reverse_lazy('api-root'), permanent=False)),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
