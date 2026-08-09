"""날짜와 관측소 정보에서 만드는 정적 특징."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(result[date_column], errors="raise")
    day_of_year = dates.dt.dayofyear.astype(float)
    result["month"] = dates.dt.month.astype(int)
    result["day_of_year"] = day_of_year.astype(int)
    result["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    result["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return result


class SeasonalStationBaseline:
    """위성이 모두 없을 때 쓰는 관측소×월 평균 fallback."""

    def __init__(self, targets: tuple[str, ...] = ("TA", "HM")) -> None:
        self.targets = targets
        self.station_month_: pd.DataFrame | None = None
        self.station_: pd.DataFrame | None = None
        self.month_: pd.DataFrame | None = None
        self.global_: dict[str, float] | None = None

    def fit(self, frame: pd.DataFrame) -> "SeasonalStationBaseline":
        required = {"STN_ID", "date", *self.targets}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"기후 기준 모델 학습 컬럼이 부족합니다: {sorted(missing)}")
        work = frame.copy()
        work["STN_ID"] = work["STN_ID"].astype(str)
        work["month"] = pd.to_datetime(work["date"]).dt.month
        self.station_month_ = (
            work.groupby(["STN_ID", "month"], as_index=False)[list(self.targets)]
            .mean()
            .rename(columns={target: f"{target}_station_month" for target in self.targets})
        )
        self.station_ = (
            work.groupby("STN_ID", as_index=False)[list(self.targets)]
            .mean()
            .rename(columns={target: f"{target}_station" for target in self.targets})
        )
        self.month_ = (
            work.groupby("month", as_index=False)[list(self.targets)]
            .mean()
            .rename(columns={target: f"{target}_month" for target in self.targets})
        )
        self.global_ = {
            target: float(pd.to_numeric(work[target], errors="coerce").mean())
            for target in self.targets
        }
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if any(
            item is None
            for item in (self.station_month_, self.station_, self.month_, self.global_)
        ):
            raise RuntimeError("SeasonalStationBaseline.fit을 먼저 실행하세요.")
        work = frame[["STN_ID", "date"]].copy()
        work["STN_ID"] = work["STN_ID"].astype(str)
        work["month"] = pd.to_datetime(work["date"]).dt.month
        work = work.merge(self.station_month_, on=["STN_ID", "month"], how="left")
        work = work.merge(self.station_, on="STN_ID", how="left")
        work = work.merge(self.month_, on="month", how="left")

        predictions: dict[str, pd.Series] = {}
        for target in self.targets:
            predictions[target] = (
                work[f"{target}_station_month"]
                .fillna(work[f"{target}_station"])
                .fillna(work[f"{target}_month"])
                .fillna(self.global_[target])
                .astype(float)
            )
        return pd.DataFrame(predictions, index=frame.index)
