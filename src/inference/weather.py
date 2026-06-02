from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


OPEN_METEO_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "terrestrial_radiation",
    "sunshine_duration",
]


@dataclass(frozen=True)
class OpenMeteoClient:
    base_url: str = "https://api.open-meteo.com/v1/forecast"
    timeout_seconds: int = 10

    def get_hourly_forecast(
        self,
        target_time: str,
        latitude: float,
        longitude: float,
    ) -> dict[str, float]:
        """Fetch target-hour forecast features from Open-Meteo and map them to project columns."""
        target_ts = pd.Timestamp(target_time).floor("h")
        start_date = target_ts.date().isoformat()
        end_date = (target_ts + timedelta(days=1)).date().isoformat()
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(OPEN_METEO_HOURLY_VARIABLES),
            "timezone": "UTC",
            "start_date": start_date,
            "end_date": end_date,
        }
        url = f"{self.base_url}?{urlencode(params)}"
        with urlopen(url, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))

        return map_open_meteo_hourly(payload, target_ts)


def map_open_meteo_hourly(payload: dict[str, Any], target_ts: pd.Timestamp) -> dict[str, float]:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ValueError("Open-Meteo response does not contain hourly forecast data.")

    times = pd.to_datetime(hourly["time"], errors="coerce")
    matches = [idx for idx, value in enumerate(times) if value == target_ts]
    if not matches:
        raise ValueError(f"Target time {target_ts.isoformat()} not found in Open-Meteo response.")
    idx = matches[0]

    raw = {
        key: _value_at(hourly, key, idx)
        for key in OPEN_METEO_HOURLY_VARIABLES
        if key in hourly
    }
    return map_weather_to_project_features(raw)


def map_weather_to_project_features(raw: dict[str, float]) -> dict[str, float]:
    """Map provider field names to the weather/radiation names used by the training data."""
    global_rad = raw.get("shortwave_radiation")
    direct_rad = raw.get("direct_radiation")
    diffuse_rad = raw.get("diffuse_radiation")
    terrestrial_rad = raw.get("terrestrial_radiation")
    humidity = raw.get("relative_humidity_2m")
    cloud_cover = raw.get("cloud_cover")
    sunshine_duration_seconds = raw.get("sunshine_duration")

    mapped = {
        "temp": raw.get("temperature_2m"),
        "relative_humidity_2m:p": humidity,
        "relative_humidity_10m:p": humidity,
        "total_cloud_cover:p": cloud_cover,
        "effective_cloud_cover:p": cloud_cover,
        "global_rad:W": global_rad,
        "global_rad_1h:Wh": global_rad,
        "direct_rad:W": direct_rad,
        "direct_rad_1h:Wh": direct_rad,
        "diffuse_rad:W": diffuse_rad,
        "diffuse_rad_1h:Wh": diffuse_rad,
        "clear_sky_rad:W": terrestrial_rad,
        "clear_sky_energy_1h:J": terrestrial_rad * 3600 if terrestrial_rad is not None else None,
        "sunshine_duration_1h:min": (
            sunshine_duration_seconds / 60 if sunshine_duration_seconds is not None else None
        ),
    }
    return {key: float(value) for key, value in mapped.items() if value is not None}


def _value_at(hourly: dict[str, Any], key: str, idx: int) -> float:
    values = hourly[key]
    value = values[idx]
    if value is None:
        raise ValueError(f"Open-Meteo field {key} is missing at index {idx}.")
    return float(value)
