from django.urls import path
from . import views

urlpatterns = [
    # Health
    path("",                  views.HealthView.as_view(),       name="health"),

    # Utility
    path("states",            views.StatesView.as_view(),       name="states"),
    path("tables",            views.TablesView.as_view(),       name="tables"),
    path("count",             views.CountView.as_view(),        name="count"),
    path("date-range",        views.DateRangeView.as_view(),    name="date-range"),

    # Read – weather
    path("weather/mean",      views.WeatherMeanView.as_view(),  name="weather-mean"),
    path("weather/loc",       views.WeatherLocView.as_view(),   name="weather-loc"),

    # Read – load / forecast / sldc
    path("load",              views.LoadView.as_view(),         name="load"),
    path("forecast",          views.ForecastView.as_view(),     name="forecast"),
    path("sldc",              views.SldcView.as_view(),         name="sldc"),
    path("sldc/forecast",     views.SldcForecastView.as_view(), name="sldc-forecast"),

    # Write
    path("upsert",            views.UpsertView.as_view(),       name="upsert"),
    path("bulk-upsert",       views.BulkUpsertView.as_view(),   name="bulk-upsert"),
]
