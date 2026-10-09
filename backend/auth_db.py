"""
auth_db.py — SQLAlchemy models and DB helpers for authentication,
sessions, forecast results, and model weights.

Uses SQLite by default (auth.db in the project root) so no extra
infrastructure is needed.  Override DB_URL via the AUTH_DB_URL env var
to point at MySQL/PostgreSQL in production.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

# ── DB connection ──────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.parent          # project root
_DEFAULT_DB = f"sqlite:///{_HERE / 'auth.db'}"
# Accounts live in the same PostgreSQL database as the pipeline data.
DB_URL = os.environ.get("AUTH_DB_URL") or os.environ.get("DATABASE_URL") or _DEFAULT_DB

engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ── ORM models ─────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    username   = Column(String(64), unique=True, nullable=False, index=True)
    email      = Column(String(128), unique=True, nullable=False, index=True)
    hashed_pw  = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    sessions          = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    forecast_results  = relationship("ForecastResult", back_populates="user", cascade="all, delete-orphan")
    model_weights     = relationship("ModelWeights",   back_populates="user", cascade="all, delete-orphan")


class UserSession(Base):
    __tablename__ = "user_sessions"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    token      = Column(String(512), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="sessions")


class ForecastResult(Base):
    """Persisted output of a completed forecast job."""
    __tablename__ = "forecast_results"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=True)   # nullable for anonymous
    job_id       = Column(String(64), unique=True, nullable=False, index=True)
    date         = Column(String(16), nullable=False)
    region       = Column(String(64), nullable=False)
    baseline_days= Column(Integer, nullable=True)
    result_json  = Column(Text, nullable=False)    # full JSON payload
    created_at   = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="forecast_results")


class ModelWeights(Base):
    """Persisted block-driver or short-term model weights."""
    __tablename__ = "model_weights"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=True)
    region       = Column(String(64), nullable=False)
    weights_type = Column(String(64), nullable=False)   # e.g. 'block_driver', 'short_term'
    weights_json = Column(Text, nullable=False)
    created_at   = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="model_weights")


# ── Create all tables on import ───────────────────────────────────────────
Base.metadata.create_all(bind=engine)


# ── Dependency helper ──────────────────────────────────────────────────────
def get_db():
    """FastAPI dependency — yields a DB session and closes it on teardown."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── CRUD helpers ───────────────────────────────────────────────────────────
def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def create_user(db: Session, username: str, email: str, hashed_pw: str) -> User:
    user = User(username=username, email=email, hashed_pw=hashed_pw)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_session(db: Session, user_id: int, token: str, ttl_hours: int = 72) -> UserSession:
    expires = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
    sess = UserSession(user_id=user_id, token=token, expires_at=expires)
    db.add(sess)
    db.commit()
    db.refresh(sess)
    return sess


def get_session(db: Session, token: str) -> Optional[UserSession]:
    return db.query(UserSession).filter(UserSession.token == token).first()


def delete_session(db: Session, token: str) -> None:
    db.query(UserSession).filter(UserSession.token == token).delete()
    db.commit()


def save_forecast_result(
    db: Session,
    job_id: str,
    date: str,
    region: str,
    baseline_days: int,
    result: dict,
    user_id: Optional[int] = None,
) -> ForecastResult:
    # Upsert: if job already saved, update; else insert
    existing = db.query(ForecastResult).filter(ForecastResult.job_id == job_id).first()
    if existing:
        existing.result_json = json.dumps(result, default=str)
        db.commit()
        db.refresh(existing)
        return existing
    fr = ForecastResult(
        user_id=user_id,
        job_id=job_id,
        date=date,
        region=region,
        baseline_days=baseline_days,
        result_json=json.dumps(result, default=str),
    )
    db.add(fr)
    db.commit()
    db.refresh(fr)
    return fr


def save_model_weights(
    db: Session,
    region: str,
    weights_type: str,
    weights: dict,
    user_id: Optional[int] = None,
) -> ModelWeights:
    mw = ModelWeights(
        user_id=user_id,
        region=region,
        weights_type=weights_type,
        weights_json=json.dumps(weights, default=str),
    )
    db.add(mw)
    db.commit()
    db.refresh(mw)
    return mw


def list_forecast_results(db: Session, user_id: Optional[int] = None, limit: int = 50):
    q = db.query(ForecastResult)
    if user_id is not None:
        q = q.filter(ForecastResult.user_id == user_id)
    return q.order_by(ForecastResult.created_at.desc()).limit(limit).all()


def list_model_weights(db: Session, user_id: Optional[int] = None, region: str = None, limit: int = 50):
    q = db.query(ModelWeights)
    if user_id is not None:
        q = q.filter(ModelWeights.user_id == user_id)
    if region:
        q = q.filter(ModelWeights.region == region)
    return q.order_by(ModelWeights.created_at.desc()).limit(limit).all()
