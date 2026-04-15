from django.urls import path
from . import views

urlpatterns = [
    # Health
    path("",                  views.HealthView.as_view(),       name="health"),

    # Frontend app API (Django-owned fast endpoints; model training stays on FastAPI)
    path("api/v2/config",          views.V2ConfigView.as_view(),         name="v2-config"),
    path("api/v2/settings",        views.V2SettingsView.as_view(),       name="v2-settings"),
    path("api/v2/reload",          views.V2ReloadView.as_view(),         name="v2-reload"),
    path("api/v2/load_benchmarks", views.V2LoadBenchmarksView.as_view(), name="v2-load-benchmarks"),
    path("api/v2/load_series",     views.V2LoadSeriesView.as_view(),     name="v2-load-series"),
    path("api/v2/load_change",     views.V2LoadChangeView.as_view(),     name="v2-load-change"),
    path("api/v2/analysis",        views.V2AnalysisView.as_view(),       name="v2-analysis"),

    # Frontend pipeline browser compatibility
    path("api/pipeline/states",             views.PipelineStatesCompatView.as_view(),     name="pipeline-states"),
    path("api/pipeline/weather/<str:state>", views.PipelineWeatherCompatView.as_view(),    name="pipeline-weather"),
    path("api/pipeline/weather-loc/<str:state>", views.PipelineWeatherLocCompatView.as_view(), name="pipeline-weather-loc"),
    path("api/pipeline/load/<str:state>",    views.PipelineLoadCompatView.as_view(),       name="pipeline-load"),
    path("api/pipeline/forecast/<str:state>", views.PipelineForecastCompatView.as_view(),  name="pipeline-forecast"),
    path("api/pipeline/similarity/<str:state>", views.PipelineSimilarityCompatView.as_view(), name="pipeline-similarity"),

    # Legacy simulator/compute tools still implemented in FastAPI, routed via Django.
    path("api/simulator/<path:path>", views.FastApiProxyView.as_view(), {"prefix": "simulator"}, name="fastapi-simulator-proxy"),
    path("api/scenario/<path:path>", views.FastApiProxyView.as_view(), {"prefix": "scenario"}, name="fastapi-scenario-proxy"),
    path("api/scenarios", views.FastApiProxyView.as_view(), {"path": "scenarios"}, name="fastapi-scenarios-proxy"),
    path("api/weather-impact/calculate", views.FastApiProxyView.as_view(), {"path": "weather-impact/calculate"}, name="fastapi-weather-impact-proxy"),
    path("api/temperature-sensitivity/calculate", views.FastApiProxyView.as_view(), {"path": "temperature-sensitivity/calculate"}, name="fastapi-temp-sensitivity-proxy"),

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
