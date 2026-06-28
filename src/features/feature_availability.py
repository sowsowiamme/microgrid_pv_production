from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


FeatureMode = Literal["forecast_proxy", "history_only"]

DETERMINISTIC_TIME_AND_SOLAR_FEATURES = {
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_daylight",
    "sun_elevation:d",
    "sun_azimuth:d",
}

HISTORICAL_PV_FEATURES = {
    "pv_change_1h",
}


@dataclass(frozen=True)
class FeatureAvailabilityReport:
    mode: FeatureMode
    kept_columns: list[str]
    removed_columns: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "kept_columns": self.kept_columns,
            "removed_columns": self.removed_columns,
        }


def apply_feature_availability(
    data: pd.DataFrame,
    mode: FeatureMode,
    target_column: str = "pv_production",
    time_column: str = "time",
) -> tuple[pd.DataFrame, FeatureAvailabilityReport]:
    """
    Filter columns according to when features would be available in production.

    forecast_proxy:
        Keeps target-hour weather columns and treats them as weather forecast proxies.
    history_only:
        Keeps only historical PV lags and deterministic time/solar-position features.
        Target-hour observed weather/radiation columns are removed before feature selection.
    """
    if mode == "forecast_proxy":
        kept_columns = list(data.columns)
        return data.copy(), FeatureAvailabilityReport(
            mode=mode,
            kept_columns=kept_columns,
            removed_columns=[],
        )

    if mode != "history_only":
        raise ValueError(f"Unsupported feature mode: {mode}")

    kept_columns = [
        column
        for column in data.columns
        if _is_history_only_available(column, target_column=target_column, time_column=time_column)
    ]
    removed_columns = [column for column in data.columns if column not in kept_columns]
    return data[kept_columns].copy(), FeatureAvailabilityReport(
        mode=mode,
        kept_columns=kept_columns,
        removed_columns=removed_columns,
    )


def _is_history_only_available(column: str, target_column: str, time_column: str) -> bool:
    if column in {target_column, time_column, "date", "Unnamed: 0"}:
        return True
    if column.startswith(f"{target_column}_lag_"):
        return True
    if column in HISTORICAL_PV_FEATURES:
        return True
    return column in DETERMINISTIC_TIME_AND_SOLAR_FEATURES
