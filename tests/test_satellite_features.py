from __future__ import annotations

import numpy as np
import pandas as pd

from gk2a_weather.features.satellite import extract_channel_features


def test_extract_channel_patch_features() -> None:
    array = np.arange(25, dtype=float).reshape(5, 5)
    array[0, 0] = np.nan
    stations = pd.DataFrame(
        {"STN_ID": [108, 159], "row": [2, 0], "col": [2, 0]}
    )
    features = extract_channel_features(
        array, stations, channel="IR105", radii_pixels=(0, 1)
    )
    assert features.loc[0, "IR105_center_mean"] == 12.0
    assert features.loc[0, "IR105_r1px_mean"] == 12.0
    assert np.isnan(features.loc[1, "IR105_center_mean"])
    assert features.loc[1, "IR105_missing"] == 1

