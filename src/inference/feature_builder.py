from __future__ import annotations

import math
from typing import Any, Iterable

import pandas as pd

from src.inference.weather import OpenMeteoClient


def build_features_for_prediction(
    feature_columns: Iterable[str],
    target_time: str,
    recent_pv: dict[str, Any],
    latitude: float | None = None,
    longitude: float | None = None,
    weather_forecast: dict[str, Any] | None = None,
    weather_client: OpenMeteoClient | None = None,
) -> dict[str, float]:
    """Construct model feature values from target time, historical PV, and weather forecast."""
    target_ts = pd.Timestamp(target_time)
    weather_values = _coerce_float_dict(weather_forecast or {})
    deterministic_values = _time_features(target_ts)
    if latitude is not None and longitude is not None:
        elevation = _solar_elevation_degrees(target_ts, latitude, longitude)
        deterministic_values["sun_elevation:d"] = elevation
        deterministic_values["is_daylight"] = 1.0 if elevation >= 2 else 0.0

    pv_values = _historical_pv_features(recent_pv)
    all_values = {**weather_values, **deterministic_values, **pv_values}

    missing_before_api = [column for column in feature_columns if column not in all_values]
    if missing_before_api and weather_forecast is None and latitude is not None and longitude is not None:
        client = weather_client or OpenMeteoClient()
        weather_values.update(client.get_hourly_forecast(target_time, latitude, longitude))
        all_values = {**weather_values, **deterministic_values, **pv_values}

    missing = [column for column in feature_columns if column not in all_values]
    if missing:
        raise ValueError(f"Could not build required feature(s): {missing}")

    return {column: float(all_values[column]) for column in feature_columns}


def build_prediction_frame(
    feature_columns: Iterable[str],
    target_time: str,
    recent_pv: dict[str, Any],
    latitude: float | None = None,
    longitude: float | None = None,
    weather_forecast: dict[str, Any] | None = None,
    weather_client: OpenMeteoClient | None = None,
) -> pd.DataFrame:
    features = build_features_for_prediction(
        feature_columns=feature_columns,
        target_time=target_time,
        recent_pv=recent_pv,
        latitude=latitude,
        longitude=longitude,
        weather_forecast=weather_forecast,
        weather_client=weather_client,
    )
    return pd.DataFrame([features], columns=list(feature_columns))


def _historical_pv_features(recent_pv: dict[str, Any]) -> dict[str, float]:
    values = _coerce_float_dict(recent_pv)
    features: dict[str, float] = {}
    for key, value in values.items():
        if key.startswith("pv_production_lag_"):
            features[key] = value
        elif key.startswith("lag_"):
            suffix = key.removeprefix("lag_")
            features[f"pv_production_lag_{suffix}"] = value

    lag_1 = features.get("pv_production_lag_1")
    lag_2 = features.get("pv_production_lag_2")
    if lag_1 is not None and lag_2 is not None:
        features["pv_change_1h"] = abs(lag_1 - lag_2) + 1

    return features


def _time_features(target_ts: pd.Timestamp) -> dict[str, float]:
    return {
        "hour": float(target_ts.hour),
        "day_of_week": float(target_ts.dayofweek),
        "month": float(target_ts.month),
        "day_of_year": float(target_ts.dayofyear),
    }


def _coerce_float_dict(values: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in values.items() if value is not None}


def _solar_elevation_degrees(target_ts: pd.Timestamp, latitude: float, longitude: float) -> float:
    """Approximate solar elevation using NOAA-style equations."""
    ts_utc = target_ts.tz_localize("UTC") if target_ts.tzinfo is None else target_ts.tz_convert("UTC")
    day_of_year = ts_utc.dayofyear
    fractional_hour = ts_utc.hour + ts_utc.minute / 60 + ts_utc.second / 3600
    gamma = 2 * math.pi / 365 * (day_of_year - 1 + (fractional_hour - 12) / 24)

    declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    true_solar_minutes = (fractional_hour * 60 + equation_of_time + 4 * longitude) % 1440
    hour_angle = math.radians(true_solar_minutes / 4 - 180)
    latitude_rad = math.radians(latitude)

    elevation_rad = math.asin(
        math.sin(latitude_rad) * math.sin(declination)
        + math.cos(latitude_rad) * math.cos(declination) * math.cos(hour_angle)
    )
    return float(math.degrees(elevation_rad))
