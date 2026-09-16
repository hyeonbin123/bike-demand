"""서비스용 테이블. 설명은 docs/serving-schema.md."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Double, Index, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Station(Base):
    __tablename__ = "stations"

    station_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    station_no: Mapped[int | None] = mapped_column(Integer)
    station_name: Mapped[str | None] = mapped_column(String(200))
    district: Mapped[str | None] = mapped_column(String(50))
    lat: Mapped[float | None] = mapped_column(Double)
    lon: Mapped[float | None] = mapped_column(Double)
    docks: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(20))  # history | realtime
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class RealtimeSnapshot(Base):
    __tablename__ = "realtime_snapshots"
    __table_args__ = (Index("ix_realtime_snapshots_fetched_at", "fetched_at"),)

    station_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    bike_count: Mapped[int] = mapped_column(Integer)
    rack_count: Mapped[int | None] = mapped_column(Integer)


class WeatherForecast(Base):
    __tablename__ = "weather_forecasts"

    base_datetime: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    fcst_datetime: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    category: Mapped[str] = mapped_column(String(10), primary_key=True)
    nx: Mapped[int] = mapped_column(Integer, primary_key=True)
    ny: Mapped[int] = mapped_column(Integer, primary_key=True)
    value: Mapped[str] = mapped_column(String(50))


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (Index("ix_predictions_hour_start", "hour_start"),)

    station_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    hour_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    model_version: Mapped[str] = mapped_column(String(100), primary_key=True)
    predicted_rentals: Mapped[float] = mapped_column(Double)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
