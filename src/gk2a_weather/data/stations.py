"""대회 station_list.csv 컬럼을 안전하게 정규화한다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


STATION_ALIASES = {
    "STN": "STN_ID",
    "stn": "STN_ID",
    "stn_id": "STN_ID",
    "LAT": "latitude",
    "lat": "latitude",
    "Latitude": "latitude",
    "LON": "longitude",
    "lon": "longitude",
    "Longitude": "longitude",
    "HT": "altitude",
    "ALT": "altitude",
    "alt": "altitude",
    "height": "altitude",
}


def load_station_list(path: str | Path) -> pd.DataFrame:
    station_path = Path(path)
    if not station_path.exists():
        raise FileNotFoundError(
            f"station_list.csv가 없습니다: {station_path}\n"
            "캐글 제공 파일을 data/metadata/station_list.csv에 넣어주세요."
        )

    frame = pd.read_csv(station_path).rename(columns=STATION_ALIASES)
    if "STN_ID" not in frame.columns:
        raise ValueError(
            f"관측소 파일에 STN_ID 컬럼이 필요합니다. 현재: {frame.columns.tolist()}"
        )
    frame["STN_ID"] = pd.to_numeric(frame["STN_ID"], errors="raise").astype(int)
    if frame["STN_ID"].duplicated().any():
        duplicated = frame.loc[frame["STN_ID"].duplicated(), "STN_ID"].tolist()
        raise ValueError(f"중복 관측소가 있습니다: {duplicated[:10]}")
    return frame.sort_values("STN_ID").reset_index(drop=True)

