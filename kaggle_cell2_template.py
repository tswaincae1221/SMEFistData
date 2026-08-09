"""공식 submission_notebook.ipynb의 2번 셀에 맞춰 옮길 템플릿.

1번 셀의 API_KEY, PRED_DATES, STATIONS를 그대로 사용합니다.
공식 1번·3번 셀은 수정하지 마세요.
"""

import glob
import sys

import joblib
import pandas as pd
import yaml


CODE_ROOT = glob.glob("/kaggle/input/**/gk2a_weather", recursive=True)[0]
sys.path.insert(0, CODE_ROOT.rsplit("/gk2a_weather", 1)[0])

from gk2a_weather.kaggle import run_kaggle_inference


MODEL_PATH = glob.glob("/kaggle/input/**/baseline.joblib", recursive=True)[0]
PIXEL_PATH = glob.glob("/kaggle/input/**/station_pixels.csv", recursive=True)[0]
CONFIG_PATH = glob.glob("/kaggle/input/**/data.yaml", recursive=True)[0]

bundle = joblib.load(MODEL_PATH)
station_pixels = pd.read_csv(PIXEL_PATH)
with open(CONFIG_PATH, "r", encoding="utf-8") as stream:
    satellite_config = yaml.safe_load(stream)["satellite"]

# ★ pred 변수명은 대회 규칙상 고정 ★
pred = run_kaggle_inference(
    api_key=API_KEY,
    pred_dates=PRED_DATES,
    stations=station_pixels[station_pixels["STN_ID"].isin(STATIONS)].copy(),
    bundle=bundle,
    satellite_config=satellite_config,
    observation_hour_kst=14,
)

