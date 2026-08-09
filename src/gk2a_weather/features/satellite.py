"""GK-2A 2차원 배열에서 관측소 주변 통계량을 추출한다.

정확한 위경도→픽셀 변환은 대회 baseline_notebook과 파일 메타데이터에
종속된다. 이 모듈은 station_list에 검증된 row/col 또는 채널별 row/col이
추가된 뒤의 특징 추출을 담당한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class CoordinateMappingError(ValueError):
    """검증된 위성 픽셀 좌표가 없을 때 발생한다."""


def _pick_2d_variable(dataset: object, channel: str) -> str:
    preferred = [channel, channel.lower(), "image_pixel_values", "image"]
    for name in preferred:
        if name in dataset.data_vars and dataset[name].ndim >= 2:
            return name
    candidates = [
        (name, variable.size)
        for name, variable in dataset.data_vars.items()
        if variable.ndim == 2 and np.issubdtype(variable.dtype, np.number)
    ]
    if not candidates:
        raise ValueError(
            f"2차원 영상 변수를 찾지 못했습니다. 변수: {list(dataset.data_vars)}"
        )
    return max(candidates, key=lambda item: item[1])[0]


def load_satellite_array(path: str | Path, channel: str) -> np.ndarray:
    """npy 또는 NetCDF/HDF 계열 파일에서 보정된 2차원 배열을 읽는다."""
    source = Path(path)
    if source.suffix.lower() == ".npy":
        array = np.load(source)
    else:
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "위성 NetCDF/HDF 파일을 읽으려면 xarray와 h5netcdf가 필요합니다. "
                "pip install -r requirements.txt를 실행하세요."
            ) from exc
        with xr.open_dataset(source, mask_and_scale=True) as dataset:
            variable = _pick_2d_variable(dataset, channel)
            array = dataset[variable].squeeze().to_numpy()
    if array.ndim != 2:
        raise ValueError(f"위성 배열은 2차원이어야 합니다. 현재 shape={array.shape}")
    return np.asarray(array, dtype=float)


def resolve_pixel_columns(stations: pd.DataFrame, channel: str) -> tuple[str, str]:
    channel_pair = (f"{channel}_row", f"{channel}_col")
    if set(channel_pair).issubset(stations.columns):
        return channel_pair
    if {"row", "col"}.issubset(stations.columns):
        return "row", "col"
    raise CoordinateMappingError(
        "station_list에 검증된 픽셀 좌표가 없습니다. "
        f"{channel}_row/{channel}_col 또는 row/col 컬럼이 필요합니다. "
        "baseline_notebook의 좌표 변환을 확인해 data/metadata/station_pixels.csv를 만드세요."
    )


def extract_channel_features(
    array: np.ndarray,
    stations: pd.DataFrame,
    *,
    channel: str,
    radii_pixels: tuple[int, ...] = (0, 2, 5),
) -> pd.DataFrame:
    row_col, col_col = resolve_pixel_columns(stations, channel)
    result = stations[["STN_ID"]].reset_index(drop=True).copy()
    height, width = array.shape

    for station_index, station in stations.reset_index(drop=True).iterrows():
        row_raw = station[row_col]
        col_raw = station[col_col]
        invalid_coordinate = not np.isfinite(row_raw) or not np.isfinite(col_raw)
        row = int(round(row_raw)) if not invalid_coordinate else -1
        col = int(round(col_raw)) if not invalid_coordinate else -1
        inside = 0 <= row < height and 0 <= col < width

        for radius in radii_pixels:
            suffix = "center" if radius == 0 else f"r{radius}px"
            if invalid_coordinate or not inside:
                values = np.array([], dtype=float)
            elif radius == 0:
                values = np.asarray([array[row, col]], dtype=float)
            else:
                row_min, row_max = max(0, row - radius), min(height, row + radius + 1)
                col_min, col_max = max(0, col - radius), min(width, col + radius + 1)
                values = array[row_min:row_max, col_min:col_max].ravel()

            finite = values[np.isfinite(values)]
            valid_ratio = float(len(finite) / len(values)) if len(values) else 0.0
            prefix = f"{channel}_{suffix}"
            result.loc[station_index, f"{prefix}_mean"] = (
                float(np.mean(finite)) if len(finite) else np.nan
            )
            result.loc[station_index, f"{prefix}_std"] = (
                float(np.std(finite)) if len(finite) else np.nan
            )
            result.loc[station_index, f"{prefix}_min"] = (
                float(np.min(finite)) if len(finite) else np.nan
            )
            result.loc[station_index, f"{prefix}_max"] = (
                float(np.max(finite)) if len(finite) else np.nan
            )
            result.loc[station_index, f"{prefix}_p10"] = (
                float(np.quantile(finite, 0.1)) if len(finite) else np.nan
            )
            result.loc[station_index, f"{prefix}_p50"] = (
                float(np.quantile(finite, 0.5)) if len(finite) else np.nan
            )
            result.loc[station_index, f"{prefix}_p90"] = (
                float(np.quantile(finite, 0.9)) if len(finite) else np.nan
            )
            result.loc[station_index, f"{prefix}_valid_ratio"] = valid_ratio

        center = result.loc[station_index, f"{channel}_center_mean"]
        largest = max(radii_pixels)
        surrounding = result.loc[station_index, f"{channel}_r{largest}px_mean"]
        result.loc[station_index, f"{channel}_center_minus_r{largest}px"] = (
            center - surrounding if np.isfinite(center) and np.isfinite(surrounding) else np.nan
        )

    result[f"{channel}_missing"] = result[f"{channel}_center_mean"].isna().astype(int)
    return result


def add_channel_differences(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    pairs = (
        ("IR105", "IR112"),
        ("IR112", "IR123"),
        ("WV063", "WV073"),
    )
    for left, right in pairs:
        left_column = f"{left}_center_mean"
        right_column = f"{right}_center_mean"
        if left_column in result and right_column in result:
            result[f"{left}_minus_{right}"] = result[left_column] - result[right_column]
    return result
