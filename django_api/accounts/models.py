"""
accounts/models.py

All auth-related tables for VidyutPragya.
managed=True → Django creates/migrates these tables in MySQL.
"""

import secrets
from datetime import datetime, timedelta

from django.contrib.auth.hashers import check_password, make_password
from django.db import models


class VPUser(models.Model):
    username   = models.CharField(max_length=64, unique=True)
    email      = models.EmailField(max_length=128, unique=True)
    password   = models.CharField(max_length=256)   # bcrypt hash
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "vp_users"
        managed  = True

    def set_password(self, raw):
        self.password = make_password(raw)

    def check_password(self, raw):
        return check_password(raw, self.password)

    def __str__(self):
        return self.username


class VPSession(models.Model):
    user       = models.ForeignKey(VPUser, on_delete=models.CASCADE, related_name="sessions")
    token      = models.CharField(max_length=128, unique=True, db_index=True)
    expires_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "vp_sessions"
        managed  = True

    @classmethod
    def create_for_user(cls, user, ttl_hours=72):
        token = secrets.token_urlsafe(48)
        expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
        return cls.objects.create(user=user, token=token, expires_at=expires_at)

    def is_valid(self):
        return datetime.utcnow() < self.expires_at

    def __str__(self):
        return f"Session({self.user.username})"


class ForecastResult(models.Model):
    user          = models.ForeignKey(VPUser, null=True, blank=True, on_delete=models.SET_NULL, related_name="forecast_results")
    job_id        = models.CharField(max_length=64, unique=True, db_index=True)
    date          = models.CharField(max_length=16)
    region        = models.CharField(max_length=64)
    baseline_days = models.IntegerField(null=True, blank=True)
    result_json   = models.JSONField()
    created_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "vp_forecast_results"
        managed  = True
        ordering = ["-created_at"]

    def __str__(self):
        return f"Forecast({self.region} {self.date})"


class ModelWeights(models.Model):
    user         = models.ForeignKey(VPUser, null=True, blank=True, on_delete=models.SET_NULL, related_name="model_weights")
    region       = models.CharField(max_length=64)
    weights_type = models.CharField(max_length=64)   # 'block_driver' | 'short_term'
    weights_json = models.JSONField()
    created_at   = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "vp_model_weights"
        managed  = True
        ordering = ["-created_at"]

    def __str__(self):
        return f"Weights({self.weights_type} {self.region})"
