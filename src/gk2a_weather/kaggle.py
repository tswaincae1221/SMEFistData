"""Kaggle 공식 노트북 2번 셀에서 호출할 동적 추론 함수."""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from gk2a_weather.data.gk2a import api_timestamp, download_gk2a_channel
from gk2a_weather.features.satellite import (
    add_channel_differences,
    extract_channel_features,
    load_satellite_array,
)
from gk2a_weather.features.static import add_calendar_features
from gk2a_weather.models.predict import predict_frame, validate_prediction_grid
from gk2a_weather.models.train import ModelBundle


LOGGER = logging.getLogger(__name__)


def _base_grid(pred_dates: list[object], stations: pd.DataFrame) -> pd.DataFrame:
    records = [
        {"date": pd.Timestamp(day).strftime("%Y-%m-%d"), "STN_ID": int(stn)}
        for day in pred_dates
        for stn in stations["STN_ID"].tolist()
    ]
    return pd.DataFrame(records).merge(stations, on="STN_ID", how="left")


def run_kaggle_inference(
    *,
    api_key: str,
    pred_dates: list[object],
    stations: pd.DataFrame,
    bundle: ModelBundle,
    satellite_config: dict[str, object],
    observation_hour_kst: int = 14,
) -> pd.DataFrame:
    """날짜를 하드코딩하지 않고 위성 다운로드부터 pred 생성까지 수행한다.

    채널 다운로드·해석·좌표 추출 중 일부가 실패하면 해당 특징을 NaN으로 두고,
    모든 위성 특징이 없는 행은 모델 bundle의 계절 기준값으로 대체한다.
    """
    if not api_key.strip():
        raise ValueError("API_KEY가 비어 있습니다.")
    stations = stations.copy()
    stations["STN_ID"] = stations["STN_ID"].astype(int)
    base = _base_grid(pred_dates, stations)
    daily_features: list[pd.DataFrame] = []
    channels = [str(value) for value in satellite_config["channels"]]
    radii = tuple(int(value) for value in satellite_config.get("patch_radii_pixels", [0, 2, 5]))

    with tempfile.TemporaryDirectory(prefix="gk2a_inference_") as tmp_dir:
        for day_value in pred_dates:
            day = pd.Timestamp(day_value)
            timestamp_kst = day.replace(
                hour=observation_hour_kst, minute=0, second=0, microsecond=0
            ).to_pydatetime()
            day_frame = stations[["STN_ID"]].copy()
            day_frame.insert(0, "date", day.strftime("%Y-%m-%d"))

            for channel in channels:
                requested = api_timestamp(
                    timestamp_kst, str(satellite_config["api_time_basis"])
                )
                raw_path = Path(tmp_dir) / f"{channel}_{requested}.bin"
                try:
                    download_gk2a_channel(
                        timestamp_kst,
                        channel,
                        api_key=api_key,
                        output_path=raw_path,
                        area=str(satellite_config.get("area", "KO")),
                        api_time_basis=str(satellite_config["api_time_basis"]),
                        timeout_seconds=float(satellite_config.get("timeout_seconds", 90)),
                        max_retries=int(satellite_config.get("max_retries", 3)),
                        retry_backoff_seconds=float(
                            satellite_config.get("retry_backoff_seconds", 2.0)
                        ),
                        minimum_file_bytes=int(
                            satellite_config.get("minimum_file_bytes", 10_000)
                        ),
                    )
                    array = load_satellite_array(raw_path, channel)
                    features = extract_channel_features(
                        array,
                        stations,
                        channel=channel,
                        radii_pixels=radii,
                    )
                    day_frame = day_frame.merge(
                        features, on="STN_ID", how="left", validate="one_to_one"
                    )
                except Exception as exc:  # 제출 중 한 채널 실패가 전체 예측을 막지 않게 한다.
                    LOGGER.warning("%s %s 특징 실패, fallback 사용 가능: %s", day.date(), channel, exc)
                finally:
                    raw_path.unlink(missing_ok=True)
            daily_features.append(day_frame)

    satellite = pd.concat(daily_features, ignore_index=True)
    features = base.merge(satellite, on=["date", "STN_ID"], how="left")
    features = add_calendar_features(features)
    features = add_channel_differences(features)
    pred = predict_frame(bundle, features)
    expected_dates = [int(pd.Timestamp(day).strftime("%Y%m%d")) for day in pred_dates]
    expected_stations = stations["STN_ID"].astype(int).tolist()
    validate_prediction_grid(pred, expected_dates, expected_stations)
    return pred

