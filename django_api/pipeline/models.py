"""
pipeline/models.py

Read-only ORM models that map to the existing MySQL tables in rishu_db.
managed=False → Django never creates/drops these tables.

Because all tables use composite primary keys (state + date + time_block),
we nominate `state` as the Django pk field.  Reads via .filter() work fine;
for UPSERT / DELETE we fall back to raw SQL through connection.cursor()
(see db_utils.py).
"""

from django.db import models


class WeatherMean(models.Model):
    state = models.CharField(max_length=64, primary_key=True)
    date = models.DateField()
    time_block = models.IntegerField()

    # Weather columns (nullable – schema may vary per deployment)
    temperature = models.FloatField(null=True, blank=True)
    humidity = models.FloatField(null=True, blank=True)
    precipitation = models.FloatField(null=True, blank=True)
    cloud_cover = models.FloatField(null=True, blank=True)
    cloud_cover_low = models.FloatField(null=True, blank=True)
    sunshine_duration = models.FloatField(null=True, blank=True)
    direct_radiation = models.FloatField(null=True, blank=True)
    wind_speed_10m = models.FloatField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = "weather_mean"
        unique_together = [("state", "date", "time_block")]
        ordering = ["date", "time_block"]


class WeatherLoc(models.Model):
    state = models.CharField(max_length=64, primary_key=True)
    date = models.DateField()
    time_block = models.IntegerField()
    location = models.CharField(max_length=128)

    temperature = models.FloatField(null=True, blank=True)
    humidity = models.FloatField(null=True, blank=True)
    precipitation = models.FloatField(null=True, blank=True)
    cloud_cover = models.FloatField(null=True, blank=True)
    cloud_cover_low = models.FloatField(null=True, blank=True)
    sunshine_duration = models.FloatField(null=True, blank=True)
    direct_radiation = models.FloatField(null=True, blank=True)
    wind_speed_10m = models.FloatField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = "weather_loc"
        unique_together = [("state", "date", "time_block", "location")]
        ordering = ["date", "time_block", "location"]


class Load(models.Model):
    state = models.CharField(max_length=64, primary_key=True)
    date = models.DateField()
    time_block = models.IntegerField()
    total_drawal = models.FloatField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = "`load`"   # backtick-escape the reserved word
        unique_together = [("state", "date", "time_block")]
        ordering = ["date", "time_block"]


class Forecast(models.Model):
    state = models.CharField(max_length=64, primary_key=True)
    date = models.DateField()
    block = models.IntegerField()
    forecast_mw = models.FloatField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = "forecast"
        unique_together = [("state", "date", "block")]
        ordering = ["date", "block"]


class Sldc(models.Model):
    state = models.CharField(max_length=64, primary_key=True)
    date = models.DateField()
    time_block = models.IntegerField()
    total_drawal = models.FloatField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = "sldc"
        unique_together = [("state", "date", "time_block")]
        ordering = ["date", "time_block"]


class SldcForecast(models.Model):
    state = models.CharField(max_length=64, primary_key=True)
    date = models.DateField()
    time_block = models.IntegerField()
    forecast_mw = models.FloatField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = "sldc_forecast"
        unique_together = [("state", "date", "time_block")]
        ordering = ["date", "time_block"]
