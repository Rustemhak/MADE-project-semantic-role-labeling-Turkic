from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('kernel/', include('kernel.urls')),
    path('about/', include('kernel.urls'), name='about'),
    path('admin/', admin.site.urls),
]