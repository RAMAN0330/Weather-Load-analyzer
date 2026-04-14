"""
accounts/views.py

Auth endpoints for VidyutPragya:
  POST /auth/register
  POST /auth/login
  POST /auth/logout
  GET  /auth/me
  POST /auth/results/save
  GET  /auth/results
  POST /auth/weights/save
  GET  /auth/weights
"""

import json
from datetime import datetime

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import ForecastResult, ModelWeights, VPSession, VPUser


# ── Auth helper ───────────────────────────────────────────────────────────────

def _get_user_from_request(request):
    """Return VPUser if Bearer token is valid, else None."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:]
    try:
        sess = VPSession.objects.select_related("user").get(token=token)
    except VPSession.DoesNotExist:
        return None
    if not sess.is_valid():
        sess.delete()
        return None
    return sess.user


def _require_auth(request):
    """Return (user, error_response). Call at start of protected views."""
    user = _get_user_from_request(request)
    if user is None:
        return None, Response({"detail": "Not authenticated"}, status=status.HTTP_401_UNAUTHORIZED)
    return user, None


# ── Register ──────────────────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name="dispatch")
class RegisterView(APIView):
    def post(self, request):
        data = request.data
        username = str(data.get("username", "")).strip()
        email    = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", ""))

        if not username or not email or not password:
            return Response({"detail": "username, email and password are required."}, status=400)
        if len(password) < 6:
            return Response({"detail": "Password must be at least 6 characters."}, status=400)
        if VPUser.objects.filter(username=username).exists():
            return Response({"detail": "Username already taken."}, status=409)
        if VPUser.objects.filter(email=email).exists():
            return Response({"detail": "Email already registered."}, status=409)

        user = VPUser(username=username, email=email)
        user.set_password(password)
        user.save()

        return Response(
            {"id": user.id, "username": user.username, "email": user.email},
            status=status.HTTP_201_CREATED,
        )


# ── Login ─────────────────────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name="dispatch")
class LoginView(APIView):
    def post(self, request):
        data     = request.data
        username = str(data.get("username", "")).strip()
        password = str(data.get("password", ""))

        try:
            user = VPUser.objects.get(username=username)
        except VPUser.DoesNotExist:
            return Response({"detail": "Invalid username or password."}, status=401)

        if not user.check_password(password):
            return Response({"detail": "Invalid username or password."}, status=401)

        sess = VPSession.create_for_user(user)
        return Response({
            "token": sess.token,
            "user": {"id": user.id, "username": user.username, "email": user.email},
        })


# ── Logout ────────────────────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name="dispatch")
class LogoutView(APIView):
    def post(self, request):
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            VPSession.objects.filter(token=auth[7:]).delete()
        return Response({"detail": "Logged out."})


# ── Me ────────────────────────────────────────────────────────────────────────

class MeView(APIView):
    def get(self, request):
        user, err = _require_auth(request)
        if err:
            return err
        return Response({"id": user.id, "username": user.username, "email": user.email})


# ── Forecast Results ──────────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name="dispatch")
class SaveResultView(APIView):
    """Called by FastAPI after a forecast job completes."""
    def post(self, request):
        data     = request.data
        job_id   = data.get("job_id")
        date     = data.get("date")
        region   = data.get("region")
        baseline = data.get("baseline_days")
        result   = data.get("result")

        if not job_id or not date or not region or result is None:
            return Response({"detail": "job_id, date, region, result required."}, status=400)

        # Optionally attach to a user if token provided
        user = _get_user_from_request(request)

        obj, created = ForecastResult.objects.update_or_create(
            job_id=job_id,
            defaults=dict(
                user=user,
                date=date,
                region=region,
                baseline_days=baseline,
                result_json=result,
            ),
        )
        return Response({"id": obj.id, "created": created}, status=201 if created else 200)


class ListResultsView(APIView):
    def get(self, request):
        user, err = _require_auth(request)
        if err:
            return err
        limit = int(request.query_params.get("limit", 50))
        rows = ForecastResult.objects.filter(user=user).order_by("-created_at")[:limit]
        return Response([
            {
                "id": r.id,
                "job_id": r.job_id,
                "date": r.date,
                "region": r.region,
                "baseline_days": r.baseline_days,
                "created_at": r.created_at.isoformat(),
                "result": r.result_json,
            }
            for r in rows
        ])


# ── Model Weights ─────────────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name="dispatch")
class SaveWeightsView(APIView):
    """Called by FastAPI after computing block driver weights."""
    def post(self, request):
        data         = request.data
        region       = data.get("region")
        weights_type = data.get("weights_type", "block_driver")
        weights      = data.get("weights")

        if not region or weights is None:
            return Response({"detail": "region and weights required."}, status=400)

        user = _get_user_from_request(request)

        obj = ModelWeights.objects.create(
            user=user,
            region=region,
            weights_type=weights_type,
            weights_json=weights,
        )
        return Response({"id": obj.id}, status=201)


class ListWeightsView(APIView):
    def get(self, request):
        user, err = _require_auth(request)
        if err:
            return err
        region = request.query_params.get("region")
        limit  = int(request.query_params.get("limit", 50))
        qs = ModelWeights.objects.filter(user=user)
        if region:
            qs = qs.filter(region=region)
        rows = qs.order_by("-created_at")[:limit]
        return Response([
            {
                "id": r.id,
                "region": r.region,
                "weights_type": r.weights_type,
                "created_at": r.created_at.isoformat(),
                "weights": r.weights_json,
            }
            for r in rows
        ])
