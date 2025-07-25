from django.urls import path
from . import views

app_name = 'dresscode_checker'

urlpatterns = [
    path('', views.home, name='home'),
    path('check/', views.check_outfit, name='check'),
    path('guidelines/', views.guidelines, name='guidelines'),
    path('about/', views.about, name='about'),
    path('api/check/', views.api_check_compliance, name='api_check'),
    path('api/stats/', views.api_stats, name='api_stats'),
    path('api/recent/', views.api_recent_checks, name='api_recent'),
]