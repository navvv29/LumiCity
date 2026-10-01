import streamlit as st
import sqlite3
import pandas as pd
import datetime
import os
import time
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import streamlit.components.v1 as components
import json
import hashlib
import threading
import cv2
import numpy as np
from ultralytics import YOLO
import requests
import pickle
from math import pi

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(_DIR, "kerala_smart_lights.db")
CREDENTIALS_FILE = os.path.join(_DIR, "credentials.json")
MODEL_PKL = os.path.join(_DIR, "intensity_model.pkl")
SCALER_PKL = os.path.join(_DIR, "intensity_scaler.pkl")
REFRESH_INTERVAL_SECONDS = 24 * 60 * 60

st.set_page_config(page_title="LumiCity Kerala", layout="wide", page_icon="💡", initial_sidebar_state="collapsed")

VALID_CITIES = {
    "TVM": "Thiruvananthapuram",
    "KCH": "Kochi",
    "KZD": "Kozhikode",
    "TCR": "Thrissur",
    "KLM": "Kollam",
    "KNR": "Kannur",
    "ALP": "Alappuzha",
    "PKD": "Palakkad",
}
COCO_WEIGHTS = {0: 1.0, 1: 1.2, 2: 2.0, 3: 1.5, 5: 3.5, 7: 3.5}
COCO_LABELS = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# ---------------------------------------------------------
# API EXTRACTORS — used by DataRefreshWorker
# ---------------------------------------------------------
def fetch_real_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": "cloudcover,visibility,weathercode",
        "timezone": "Asia/Kolkata",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
        df = pd.DataFrame({
            "time": pd.to_datetime(data["hourly"]["time"]),
            "cloudcover": data["hourly"]["cloudcover"],
            "visibility": data["hourly"]["visibility"],
        })
        df["visibility"] = df["visibility"].ffill().bfill().fillna(10000.0)
        return df
    except Exception:
        return pd.DataFrame()

_viirs_cache = {}

def fetch_nasa_viirs_data(lat, lon, date_str):
    if date_str in _viirs_cache:
        return _viirs_cache[date_str]
    bbox = [lat - 0.05, lon - 0.05, lat + 0.05, lon + 0.05]
    search_url = (
        f"https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details?"
        f"product=VNP46A1&collection=5000&dateRanges={date_str}..{date_str}"
        f"&coordsOrTiles=coords&coordinates=[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"
    )
    try:
        res = requests.get(search_url, timeout=15, headers={"Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6Im5hdnZ2MjkiLCJleHAiOjE3NzkwNDM1NzcsImlhdCI6MTc3Mzg1OTU3NywiaXNzIjoiaHR0cHM6Ly91cnMuZWFydGhkYXRhLm5hc2EuZ292IiwiaWRlbnRpdHlfcHJvdmlkZXIiOiJlZGxfb3BzIiwiYWNyIjoiZWRsIiwiYXNzdXJhbmNlX2xldmVsIjozfQ.38JuuyZmgMG184hv3vMq6kgX5mT6atrZfUhUrJfj0ED54In0u0V2IpOy1vvCBeYm04d8vbVrgMFfGJMOr1rRt1sBsmqSa9mhITyVnNsPYVORNChh50HNRSEUfgLAumcOOqrSFriAOCyT4tXUkIps0lCWAObjrokMC1nivIrq47gjFvGbVI2otnDRqwYDoSIOj9lKcTw76PNAHCbAWcXKmoNUL2uITK3hYkOo0nnBIko0GMfnu1lYHUStPQ7NSPpGl3dwqNw1jD5fca9m0rbMowQZ2mbypT4aMk4yik81sCDNM-HZULY6Yevbm6E2Q2r2oiPnfPhN11PC1IacrryNGg"}).json()
        if 'content' in res and len(res['content']) > 0:
            file_url = res['content'][0].get('downloadsLink', '')
            session = requests.Session()
            r = session.get(file_url, stream=True, timeout=60, headers={"Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6Im5hdnZ2MjkiLCJleHAiOjE3NzkwNDM1NzcsImlhdCI6MTc3Mzg1OTU3NywiaXNzIjoiaHR0cHM6Ly91cnMuZWFydGhkYXRhLm5hc2EuZ292IiwiaWRlbnRpdHlfcHJvdmlkZXIiOiJlZGxfb3BzIiwiYWNyIjoiZWRsIiwiYXNzdXJhbmNlX2xldmVsIjozfQ.38JuuyZmgMG184hv3vMq6kgX5mT6atrZfUhUrJfj0ED54In0u0V2IpOy1vvCBeYm04d8vbVrgMFfGJMOr1rRt1sBsmqSa9mhITyVnNsPYVORNChh50HNRSEUfgLAumcOOqrSFriAOCyT4tXUkIps0lCWAObjrokMC1nivIrq47gjFvGbVI2otnDRqwYDoSIOj9lKcTw76PNAHCbAWcXKmoNUL2uITK3hYkOo0nnBIko0GMfnu1lYHUStPQ7NSPpGl3dwqNw1jD5fca9m0rbMowQZ2mbypT4aMk4yik81sCDNM-HZULY6Yevbm6E2Q2r2oiPnfPhN11PC1IacrryNGg"})
            if r.status_code == 200:
                import h5py, tempfile
                filepath = os.path.join(tempfile.gettempdir(), f"viirs_{date_str}.h5")
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                try:
                    with h5py.File(filepath, "r") as f:
                        rad = f["HDFEOS"]["GRIDS"]["VNP_Grid_DNB"]["Data Fields"][
                            "DNB_At_Sensor_Radiance_500m"][:]
                        valid_rad = rad[(rad > 0) & (rad < 65535)]
                        if len(valid_rad) > 0:
                            ret_val = float(np.mean(valid_rad))
                            _viirs_cache[date_str] = ret_val
                            os.remove(filepath)
                            return ret_val
                    os.remove(filepath)
                except Exception:
                    if os.path.exists(filepath):
                        os.remove(filepath)
    except Exception:
        pass
    _viirs_cache[date_str] = 15.4
    return 15.4

def get_traffic_factor(hour):
    return max(0.1, np.sin(pi * (hour - 4) / 12) + 0.5 * np.cos(pi * hour / 6))

# ---------------------------------------------------------
# CSS — Modern Glassmorphism Theme
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(160deg, #0a0e17 0%, #0d1321 40%, #111827 100%);
    }
    .block-container { padding-top: 4rem; }
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu {visibility: hidden;}
    .stAppDeployButton {display: none;}

    /* Glass Cards */
    .glass-card {
        background: rgba(17, 25, 40, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(0, 229, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 229, 255, 0.08);
    }

    /* Intensity Card */
    .intensity-card {
        background: rgba(17, 25, 40, 0.8);
        backdrop-filter: blur(12px);
        padding: 18px 14px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        text-align: center;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    .intensity-card:hover {
        border-color: rgba(0, 229, 255, 0.35);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 229, 255, 0.1);
    }
    .big-intensity {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00e5ff, #00b8d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        font-family: 'Inter', sans-serif;
    }
    .big-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 4px;
        font-family: 'Inter', sans-serif;
    }
    .node-badge {
        background: rgba(0, 229, 255, 0.1);
        padding: 4px 14px;
        border-radius: 20px;
        color: #00e5ff;
        font-weight: 600;
        font-size: 0.75rem;
        display: inline-block;
        margin-bottom: 8px;
        border: 1px solid rgba(0, 229, 255, 0.2);
        letter-spacing: 0.5px;
    }

    /* TRS Pill */
    .trs-pill {
        background: rgba(27, 94, 32, 0.3);
        padding: 5px 14px;
        border-radius: 20px;
        color: #66bb6a;
        font-weight: 700;
        display: inline-block;
        border: 1px solid rgba(102, 187, 106, 0.25);
        font-size: 0.85rem;
    }

    /* Camera Result */
    .cam-result-card {
        background: rgba(13, 27, 42, 0.85);
        backdrop-filter: blur(16px);
        padding: 28px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        text-align: center;
    }
    .cam-big {
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00e5ff 0%, #40c4ff 50%, #00b0ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
        font-family: 'Inter', sans-serif;
    }

    /* Override Badge */
    .override-active {
        background: rgba(156, 39, 176, 0.15);
        padding: 8px 12px;
        border-radius: 10px;
        border: 1px solid rgba(156, 39, 176, 0.3);
        color: #ce93d8;
        text-align: center;
        margin-top: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Header Bar */
    .header-bar {
        background: rgba(17, 25, 40, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 10px 20px;
        margin-bottom: 4px;
    }

    /* Status Indicator */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Login Page */
    .login-container {
        background: rgba(17, 25, 40, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 40px 36px;
        max-width: 440px;
        margin: 60px auto;
    }
    .login-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00e5ff, #40c4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 4px;
        font-family: 'Inter', sans-serif;
    }
    .login-subtitle {
        text-align: center;
        color: rgba(255,255,255,0.4);
        font-size: 0.9rem;
        margin-bottom: 24px;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(17, 25, 40, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
    }
    .metric-label {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.4);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 2px;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(17, 25, 40, 0.5);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 229, 255, 0.1);
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.06) !important; }

    /* Dataframe */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_yolo():
    return YOLO(os.path.join(_DIR, 'yolov8n.pt'))


class CameraWorker:
    """Background camera reader to keep Streamlit UI responsive."""

    def __init__(self, camera_index=0, width=640, height=480, infer_every=4):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.infer_every = max(1, infer_every)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._state = {
            "frame_rgb": None,
            "trs": 0.0,
            "boost": 0.0,
            "counts": {},
            "error": None,
            "last_update": 0.0,
        }

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout=1.5):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self):
        with self._lock:
            return {
                "frame_rgb": self._state["frame_rgb"],
                "trs": self._state["trs"],
                "boost": self._state["boost"],
                "counts": dict(self._state["counts"]),
                "error": self._state["error"],
                "last_update": self._state["last_update"],
            }

    def _run(self):
        cap = None
        try:
            model = load_yolo()
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            if not cap.isOpened():
                with self._lock:
                    self._state["error"] = "Cannot access camera."
                return

            frame_idx = 0
            trs = 0.0
            boost = 0.0
            det_counts = {}

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.03)
                    continue

                frame_idx += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if frame_idx % self.infer_every == 0:
                    results = model(frame, conf=0.40, iou=0.45, verbose=False)[0]
                    trs = 0.0
                    det_counts = {}
                    for box in results.boxes:
                        cls = int(box.cls[0].item())
                        if cls in COCO_WEIGHTS:
                            label = COCO_LABELS[cls]
                            det_counts[label] = det_counts.get(label, 0) + 1
                            trs += COCO_WEIGHTS[cls]
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if trs >= 15.0:
                        boost = 40.0
                    elif trs >= 10.0:
                        boost = 30.0
                    elif trs >= 5.0:
                        boost = 20.0
                    elif trs >= 2.0:
                        boost = 10.0
                    else:
                        boost = 0.0

                with self._lock:
                    self._state["frame_rgb"] = frame_rgb
                    self._state["trs"] = trs
                    self._state["boost"] = boost
                    self._state["counts"] = det_counts
                    self._state["error"] = None
                    self._state["last_update"] = time.time()
        except Exception as exc:
            with self._lock:
                self._state["error"] = f"Camera worker error: {exc}"
        finally:
            if cap is not None:
                cap.release()

# ---------------------------------------------------------
# DATA REFRESH WORKER — weather every 5h, predictions every 24h
# ---------------------------------------------------------
WEATHER_REFRESH_SECONDS = 5 * 60 * 60  # 5 hours

class DataRefreshWorker:
    """Background worker that refreshes weather every 5h and
    regenerates full predictions every 24h."""

    def __init__(self, interval=REFRESH_INTERVAL_SECONDS, weather_interval=WEATHER_REFRESH_SECONDS):
        self.interval = interval
        self.weather_interval = weather_interval
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._state = {
            "last_refresh": None,
            "last_weather": None,
            "status": "IDLE",
            "error": None,
            "predictions_count": 0,
        }
        self._init_last_refresh_from_db()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout=5.0):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def _run(self):
        # On startup: refresh stale data immediately
        if self._predictions_are_stale():
            self._do_full_refresh()
        elif self._weather_is_stale():
            self._do_weather_refresh()

        # Main loop: check every 60 seconds if anything needs refreshing
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=60.0)
            if self._stop_event.is_set():
                return
            if self._predictions_are_stale():
                self._do_full_refresh()
            elif self._weather_is_stale():
                self._do_weather_refresh()

    def _init_last_refresh_from_db(self):
        try:
            conn = sqlite3.connect(DB_NAME, timeout=10)
            cur = conn.cursor()
            cur.execute("SELECT prediction_made_at FROM predictions ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            conn.close()
            if row is not None:
                made_at = datetime.datetime.fromisoformat(str(row[0]))
                with self._lock:
                    self._state["last_refresh"] = made_at
                    self._state["status"] = "OK"
        except Exception:
            pass
        # Check last weather fetch
        try:
            conn = sqlite3.connect(DB_NAME, timeout=10)
            cur = conn.cursor()
            cur.execute("SELECT fetched_at FROM weather_data ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            conn.close()
            if row is not None:
                with self._lock:
                    self._state["last_weather"] = datetime.datetime.fromisoformat(str(row[0]))
        except Exception:
            pass

    def _predictions_are_stale(self):
        try:
            conn = sqlite3.connect(DB_NAME, timeout=10)
            cur = conn.cursor()
            cur.execute("SELECT prediction_made_at FROM predictions ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            conn.close()
            if row is None:
                return True
            made_at = datetime.datetime.fromisoformat(str(row[0]))
            age = (datetime.datetime.now() - made_at).total_seconds()
            return age > self.interval
        except Exception:
            return True

    def _weather_is_stale(self):
        with self._lock:
            last_wx = self._state.get("last_weather")
        if last_wx is None:
            return True
        age = (datetime.datetime.now() - last_wx).total_seconds()
        return age > self.weather_interval

    def _get_lights_and_celebrations(self):
        conn = sqlite3.connect(DB_NAME, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        lights = pd.read_sql("SELECT light_id, city_code, latitude, longitude FROM street_lights", conn)
        celebrations = pd.read_sql("SELECT event_date, multiplier, city_code FROM celebration_calendar", conn)
        conn.close()
        return lights, celebrations

    def _do_weather_refresh(self):
        """Fetch fresh weather data only (every 5 hours)."""
        with self._lock:
            prev_status = self._state["status"]
            self._state["status"] = "RUNNING"
        try:
            lights, _ = self._get_lights_and_celebrations()
            if lights.empty:
                return
            ref_lat = lights.iloc[0]["latitude"]
            ref_lon = lights.iloc[0]["longitude"]
            now = datetime.datetime.now()
            target_date_str = now.strftime("%Y-%m-%d")

            weather_today = fetch_real_weather(ref_lat, ref_lon, target_date_str, target_date_str)
            self._populate_weather_data(weather_today, lights, target_date_str)

            with self._lock:
                self._state["last_weather"] = now
                self._state["status"] = "OK"

            try:
                _log_conn = sqlite3.connect(DB_NAME, timeout=10)
                _log_conn.execute("PRAGMA journal_mode=WAL")
                _log_conn.cursor().execute(
                    "INSERT INTO system_log (ts, level, component, message) VALUES (?, ?, ?, ?)",
                    (str(now), "INFO", "DataRefreshWorker", "Weather data refreshed (5h cycle)"))
                _log_conn.commit()
                _log_conn.close()
            except Exception:
                pass

        except Exception as exc:
            with self._lock:
                self._state["status"] = prev_status if prev_status != "RUNNING" else "ERROR"
                self._state["error"] = f"Weather refresh: {exc}"

    def _do_full_refresh(self):
        """Full refresh: weather + VIIRS + predictions (every 24 hours)."""
        with self._lock:
            self._state["status"] = "RUNNING"
            self._state["error"] = None
        try:
            with open(MODEL_PKL, "rb") as f:
                model = pickle.load(f)
            with open(SCALER_PKL, "rb") as f:
                scaler = pickle.load(f)

            now = datetime.datetime.now()
            target_date_str = now.strftime("%Y-%m-%d")

            lights, celebrations = self._get_lights_and_celebrations()
            if lights.empty:
                raise RuntimeError("No street lights found in database")

            ref_lat = lights.iloc[0]["latitude"]
            ref_lon = lights.iloc[0]["longitude"]
            try:
                weather_today = fetch_real_weather(ref_lat, ref_lon, target_date_str, target_date_str)
            except Exception:
                weather_today = pd.DataFrame()

            self._populate_weather_data(weather_today, lights, target_date_str)

            prediction_rows = []
            for _, light in lights.iterrows():
                viirs_rad = fetch_nasa_viirs_data(light["latitude"], light["longitude"], target_date_str)
                cel_mult = 1.0
                if not celebrations.empty:
                    c_rows = celebrations[
                        (celebrations["event_date"] == target_date_str)
                        & ((celebrations["city_code"] == light["city_code"])
                           | (celebrations["city_code"] == "ALL"))
                    ]
                    if not c_rows.empty:
                        cel_mult = c_rows["multiplier"].max()

                for h_offset in range(24):
                    pred_hour = (now.hour + h_offset) % 24
                    target_ts = now.replace(hour=pred_hour, minute=0, second=0) + datetime.timedelta(
                        days=h_offset // 24)
                    is_poor_vis = 0
                    if not weather_today.empty:
                        match_w = weather_today[weather_today["time"].dt.hour == pred_hour]
                        if not match_w.empty and match_w.iloc[0]["visibility"] < 3000:
                            is_poor_vis = 1

                    traffic_p = viirs_rad * get_traffic_factor(pred_hour)
                    feat = np.array([[
                        pred_hour, target_ts.weekday(),
                        1 if target_ts.weekday() >= 5 else 0,
                        target_ts.month, is_poor_vis, viirs_rad,
                        cel_mult, traffic_p,
                        light["latitude"], light["longitude"],
                    ]])
                    pred_intensity = float(np.clip(model.predict(scaler.transform(feat))[0], 0, 100))
                    prediction_rows.append((
                        str(now), str(target_ts), light["light_id"],
                        pred_intensity, max(0, pred_intensity - 10),
                        min(100, pred_intensity + 10),
                        "WEEKEND" if target_ts.weekday() >= 5 else "WEEKDAY",
                        cel_mult, is_poor_vis, "AutoRefresh-v1",
                    ))

            conn = sqlite3.connect(DB_NAME, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            cur = conn.cursor()
            cur.execute("DELETE FROM predictions")
            cur.executemany(
                """INSERT INTO predictions
                   (prediction_made_at, target_timestamp, light_id, predicted_intensity,
                    confidence_low, confidence_high, day_type, celebration_multiplier,
                    weather_override, model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                prediction_rows,
            )
            conn.commit()
            conn.close()

            try:
                _log_conn = sqlite3.connect(DB_NAME, timeout=10)
                _log_conn.execute("PRAGMA journal_mode=WAL")
                _log_conn.cursor().execute(
                    "INSERT INTO system_log (ts, level, component, message) VALUES (?, ?, ?, ?)",
                    (str(now), "INFO", "DataRefreshWorker",
                     f"Full refresh: {len(prediction_rows)} predictions + weather from live API"))
                _log_conn.commit()
                _log_conn.close()
            except Exception:
                pass

            with self._lock:
                self._state["last_refresh"] = now
                self._state["last_weather"] = now
                self._state["status"] = "OK"
                self._state["predictions_count"] = len(prediction_rows)
                self._state["error"] = None

        except Exception as exc:
            with self._lock:
                self._state["status"] = "ERROR"
                self._state["error"] = str(exc)
            try:
                _log_conn = sqlite3.connect(DB_NAME, timeout=10)
                _log_conn.cursor().execute(
                    "INSERT INTO system_log (ts, level, component, message) VALUES (?, ?, ?, ?)",
                    (str(datetime.datetime.now()), "ERROR", "DataRefreshWorker", str(exc)))
                _log_conn.commit()
                _log_conn.close()
            except Exception:
                pass

    def _populate_weather_data(self, weather_df, lights, date_str):
        if weather_df.empty:
            return
        city_codes = lights["city_code"].unique()
        now_str = str(datetime.datetime.now())
        conn = sqlite3.connect(DB_NAME, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        cur = conn.cursor()
        cur.execute("DELETE FROM weather_data WHERE timestamp LIKE ?", (f"{date_str}%",))
        for cc in city_codes:
            for _, row in weather_df.iterrows():
                vis = row["visibility"]
                cc_val = row.get("cloudcover", 0)
                is_poor = 1 if vis < 3000 else 0
                cur.execute(
                    """INSERT INTO weather_data
                       (city_code, timestamp, cloudcover, visibility,
                        is_poor_visibility, fetched_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (cc, str(row["time"]), cc_val, vis, is_poor, now_str))
        conn.commit()
        conn.close()

# ---------------------------------------------------------
# DB
# ---------------------------------------------------------
def query_db(q, params=()):
    try:
        conn = sqlite3.connect(DB_NAME, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        df = pd.read_sql_query(q, conn, params=params)
        conn.close()
        return df
    except:
        return pd.DataFrame()

def insert_db(q, params=()):
    try:
        conn = sqlite3.connect(DB_NAME, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.cursor().execute(q, params)
        conn.commit()
        conn.close()
    except:
        pass

def load_credentials():
    try:
        with open(CREDENTIALS_FILE, 'r') as f: return json.load(f)
    except: return {}

# ---------------------------------------------------------
# START DATA REFRESH WORKER (singleton via cache_resource)
# ---------------------------------------------------------
@st.cache_resource
def get_data_refresh_worker():
    worker = DataRefreshWorker(interval=REFRESH_INTERVAL_SECONDS)
    worker.start()
    return worker

_refresh_worker = get_data_refresh_worker()

# ---------------------------------------------------------
# SMART INTENSITY — minute-level interpolation
# ---------------------------------------------------------
def compute_base_intensity(light_id, sim_hour, sim_minute=0, sim_date=None, city_code=None):
    """Compute required intensity using smooth minute-level interpolation,
    weather overrides, AI predictions, and celebration multipliers.
    Daytime = 0% unless poor visibility. Night = smoothly interpolated."""

    t = sim_hour + sim_minute / 60.0
    anchors = [
        (0.0, 40), (0.5, 32), (1.0, 20), (3.0, 18), (5.0, 20), (5.5, 10), (6.0, 0),
        (17.5, 0), (18.0, 40), (18.25, 48), (18.5, 55), (19.0, 70), (19.5, 69),
        (20.0, 66), (20.5, 62), (21.0, 55), (21.5, 50), (22.0, 48),
        (22.5, 44), (23.0, 40), (23.5, 38), (24.0, 40)
    ]
    base = 0.0
    for i in range(len(anchors) - 1):
        t0, v0 = anchors[i]
        t1, v1 = anchors[i + 1]
        if t0 <= t < t1:
            frac = (t - t0) / (t1 - t0) if t1 != t0 else 0
            base = v0 + (v1 - v0) * frac
            break
    else:
        base = anchors[-1][1]

    wx = query_db("SELECT is_poor_visibility FROM weather_data WHERE city_code = ? ORDER BY timestamp DESC LIMIT 1",
                  (city_code or '',))
    is_poor_vis = False
    if not wx.empty:
        try:
            is_poor_vis = bool(wx.iloc[0]['is_poor_visibility'])
            if is_poor_vis:
                base = max(base, 60.0)
        except: pass

    p = query_db(
        "SELECT predicted_intensity FROM predictions WHERE light_id = ? AND CAST(strftime('%%H', target_timestamp) AS INTEGER) = ? ORDER BY target_timestamp DESC LIMIT 1",
        (light_id, sim_hour))
    if not p.empty and p.iloc[0]['predicted_intensity'] > 0:
        ai_val = float(p.iloc[0]['predicted_intensity'])
        if 6 <= sim_hour < 18:
            if is_poor_vis:
                base = max(base, ai_val)
        else:
            base = 0.7 * ai_val + 0.3 * base

    check_date = sim_date or datetime.date.today()
    cc = city_code or (light_id.split('-')[1] if '-' in light_id else '')
    events = query_db("SELECT multiplier FROM celebration_calendar WHERE event_date = ? AND (city_code = ? OR city_code = 'ALL')",
                      (str(check_date), cc))
    if not events.empty:
        max_mult = events['multiplier'].max()
        base = min(100.0, base * max_mult)

    return round(base, 1)

# ---------------------------------------------------------
# AUTH — Modern Login Page
# ---------------------------------------------------------
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["city_code"] = None
    if not st.session_state["authenticated"]:
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.markdown("""
            <div class='login-container'>
                <div class='login-title'>LumiCity</div>
                <div class='login-subtitle'>Kerala Smart City Street Light Control</div>
            </div>
            """, unsafe_allow_html=True)
            city_sel = st.selectbox("Select Municipality", [f"{k} - {v}" for k, v in VALID_CITIES.items()])
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            if st.button("Sign In", use_container_width=True, type="primary"):
                creds = load_credentials()
                hashed = hashlib.sha256(password.encode()).hexdigest()
                if username in creds and creds[username] == hashed:
                    st.session_state["authenticated"] = True
                    st.session_state["city_code"] = city_sel.split(" - ")[0]
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
        return False
    return True

if not check_password():
    st.stop()

# Admin can switch to any city
if st.session_state.get("username") == "admin":
    _admin_cities = [f"{k} - {v}" for k, v in VALID_CITIES.items()]
    _cur_idx = list(VALID_CITIES.keys()).index(st.session_state["city_code"]) if st.session_state["city_code"] in VALID_CITIES else 0
    _admin_sel = st.selectbox("Switch City (Admin)", _admin_cities, index=_cur_idx, key="admin_city_switch")
    st.session_state["city_code"] = _admin_sel.split(" - ")[0]

city = st.session_state["city_code"]
city_name = VALID_CITIES[city]
now = datetime.datetime.now()
hour = now.hour
minute = now.minute

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "overrides" not in st.session_state:
    st.session_state["overrides"] = {}

# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------
lights_df = query_db("SELECT * FROM street_lights WHERE city_code = ?", (city,))
zone_df = query_db("SELECT * FROM city_zones WHERE city_code = ?", (city,))

model_df = query_db("SELECT * FROM model_metrics ORDER BY trained_at DESC LIMIT 1")

wx_status = "CLEAR"
wx_status_color = "#4CAF50"
wx_df = query_db("SELECT * FROM weather_data WHERE city_code = ? ORDER BY timestamp DESC LIMIT 1", (city,))
if not wx_df.empty:
    try:
        if wx_df.iloc[0]['is_poor_visibility']:
            wx_status = "OVERRIDE"
            wx_status_color = "#f44336"
    except: pass

# Compute intensities
light_intensities = {}
for _, l in lights_df.iterrows():
    lid = l['light_id']
    ov = st.session_state["overrides"].get(lid)
    if ov and ov["mode"] == "MANUAL":
        light_intensities[lid] = float(ov["value"])
    elif ov and ov["mode"] == "OFF":
        light_intensities[lid] = 0.0
    else:
        light_intensities[lid] = compute_base_intensity(lid, hour, minute, city_code=city)

# Compute skyglow
_avg_intensity = sum(light_intensities.values()) / max(len(light_intensities), 1)
_rad_df = query_db(
    "SELECT AVG(v.raw_radiance) as avg_rad FROM viirs_data v "
    "JOIN street_lights s ON v.light_id = s.light_id WHERE s.city_code = ?", (city,))
_avg_radiance = (_rad_df.iloc[0]['avg_rad']
                 if not _rad_df.empty and pd.notnull(_rad_df.iloc[0]['avg_rad'])
                 else 5.0)
sky_val = _avg_radiance * (_avg_intensity / 100.0)
sky_str = f"{sky_val:.2f}"
sky_color = "#f44336" if sky_val > 1.5 else ("#ff9800" if sky_val > 0.8 else "#4CAF50")

# ---------------------------------------------------------
# HEADER — Modern Dashboard Header
# ---------------------------------------------------------
h1, h2 = st.columns([7, 1])
with h1:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px;">
        <div style="font-size: 1.5rem; font-weight: 800; color: white; font-family: 'Inter', sans-serif;">
            {city_name}
        </div>
        <div style="background: rgba(0,229,255,0.1); padding: 3px 12px; border-radius: 20px; border: 1px solid rgba(0,229,255,0.2);">
            <span style="color: #00e5ff; font-size: 0.75rem; font-weight: 600; letter-spacing: 1px;">SMART CITY</span>
        </div>
    </div>""", unsafe_allow_html=True)
with h2:
    if st.button("Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["overrides"] = {}
        st.rerun()

# Status bar — compute refresh status
_refresh_snap = _refresh_worker.snapshot()
_refresh_status_text = "Never"
_refresh_dot_color = "#ff9800"
if _refresh_snap["last_refresh"]:
    _refresh_ago = (datetime.datetime.now() - _refresh_snap["last_refresh"]).total_seconds()
    if _refresh_ago < 3600:
        _refresh_status_text = f"{int(_refresh_ago // 60)}m ago"
    else:
        _refresh_status_text = _refresh_snap["last_refresh"].strftime("%H:%M")
    _refresh_dot_color = "#4CAF50"
if _refresh_snap["status"] == "RUNNING":
    _refresh_status_text = "Syncing..."
    _refresh_dot_color = "#00e5ff"
elif _refresh_snap["status"] == "ERROR":
    _refresh_status_text = "Error"
    _refresh_dot_color = "#f44336"

# Weather refresh status (5h cycle)
_wx_refresh_text = "Never"
_wx_refresh_dot = "#ff9800"
if _refresh_snap["last_weather"]:
    _wx_ago = (datetime.datetime.now() - _refresh_snap["last_weather"]).total_seconds()
    if _wx_ago < 3600:
        _wx_refresh_text = f"{int(_wx_ago // 60)}m ago"
    elif _wx_ago < 18000:
        _wx_refresh_text = f"{_wx_ago / 3600:.1f}h ago"
    else:
        _wx_refresh_text = _refresh_snap["last_weather"].strftime("%H:%M")
    _wx_refresh_dot = "#4CAF50"
if _refresh_snap["status"] == "RUNNING":
    _wx_refresh_text = "Updating..."
    _wx_refresh_dot = "#00e5ff"

components.html(f"""
<div style="background:rgba(17,25,40,0.6);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.05);
            border-radius:12px;padding:10px 20px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;font-family:'Inter',system-ui,sans-serif;">
    <div style="display:flex;align-items:center;gap:6px;">
        <span id="clk" style="font-family:monospace;color:#00e5ff;font-size:0.9rem;font-weight:600;"></span>
    </div>
    <div style="display:flex;gap:20px;align-items:center;font-size:0.82rem;">
        <span style="color:#8b949e;">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#4CAF50;margin-right:6px;animation:pulse 2s infinite;"></span>VIIRS Sync OK</span>
        <span style="color:#8b949e;">Skyglow: <span style="color:{sky_color};font-weight:700;">{sky_str}</span></span>
        <span style="color:#8b949e;">Nodes: <span style="color:#00e5ff;font-weight:700;">{len(lights_df)}</span></span>
        <span style="color:#8b949e;">Weather:
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{wx_status_color};margin:0 4px;animation:pulse 2s infinite;"></span>
            <span style="color:{wx_status_color};font-weight:600;">{wx_status}</span></span>
        <span style="color:#8b949e;">WX Sync:
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{_wx_refresh_dot};margin:0 4px;animation:pulse 2s infinite;"></span>
            <span style="color:{_wx_refresh_dot};font-weight:600;">{_wx_refresh_text}</span></span>
        <span style="color:#8b949e;">Predict:
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{_refresh_dot_color};margin:0 4px;animation:pulse 2s infinite;"></span>
            <span style="color:{_refresh_dot_color};font-weight:600;">{_refresh_status_text}</span></span>
    </div>
</div>
<style>@keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.4}} }}</style>
<script>
function t(){{var n=new Date();
document.getElementById('clk').textContent=
String(n.getHours()).padStart(2,'0')+':'+String(n.getMinutes()).padStart(2,'0')+':'+String(n.getSeconds()).padStart(2,'0')+' IST';}}
setInterval(t,1000);t();
</script>
""", height=52)

st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TIME SIMULATION
# ---------------------------------------------------------
with st.expander("Time Simulation", expanded=False):
    sim_col1, sim_col2, sim_col3, sim_col4 = st.columns([1, 1.5, 0.5, 1])
    with sim_col1:
        use_sim = st.checkbox("Enable Simulation", key="time_sim")
    with sim_col2:
        sim_hour = st.slider("Hour (0-23)", 0, 23, hour, key="sim_h", disabled=not use_sim)
    with sim_col3:
        sim_min = st.slider("Minute", 0, 59, minute, key="sim_min", disabled=not use_sim)
    with sim_col4:
        sim_date = st.date_input("Date", value=now.date(), key="sim_d", disabled=not use_sim)

display_hour = sim_hour if use_sim else hour
display_minute = sim_min if use_sim else minute
display_date = sim_date if use_sim else now.date()

if use_sim:
    for _, l in lights_df.iterrows():
        lid = l['light_id']
        ov = st.session_state["overrides"].get(lid)
        if ov and ov["mode"] == "MANUAL":
            light_intensities[lid] = float(ov["value"])
        elif ov and ov["mode"] == "OFF":
            light_intensities[lid] = 0.0
        else:
            light_intensities[lid] = compute_base_intensity(lid, display_hour, display_minute, display_date, city_code=city)
    avg_sim = sum(light_intensities.values()) / max(len(light_intensities), 1)
    st.info(f"Simulating **{display_hour}:{display_minute:02d}** on **{display_date}** — Avg intensity: **{avg_sim:.1f}%**")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["City Map & Intensity", "Camera Surveillance", "Image Analysis", "Analytics", "Data & Logs"])

# =========================================================
# TAB 1: MAP + INTENSITY
# =========================================================
with tab1:
    map_col, info_col = st.columns([3, 2])
    with map_col:
        st.markdown("#### Street Light Network")
        if not zone_df.empty:
            z = zone_df.iloc[0]
            m = folium.Map(location=[z['center_lat'], z['center_lon']], zoom_start=13, tiles='CartoDB dark_matter')
            folium.Rectangle(bounds=[[z['south_lat'], z['west_lon']], [z['north_lat'], z['east_lon']]],
                             color='#1e88e5', fill=True, fill_opacity=0.06, weight=2).add_to(m)
            for _, l in lights_df.iterrows():
                lid = l['light_id']
                iv = light_intensities.get(lid, 0)
                ov = st.session_state["overrides"].get(lid)
                c = '#4CAF50' if iv < 40 else ('#ff9800' if iv < 70 else '#f44336')
                if ov: c = '#9c27b0'
                ov_tag = f"<br><b style='color:#ce93d8;'>OVERRIDE: {ov['mode']}</b>" if ov else ""
                folium.CircleMarker(
                    location=[l['latitude'], l['longitude']], radius=8,
                    popup=folium.Popup(f"<b>{l['location_name']}</b><br>ID: {lid}<br><b>Intensity: {iv:.1f}%</b>{ov_tag}", max_width=220),
                    color=c, fill=True, fill_color=c, fill_opacity=0.85
                ).add_to(m)
            st_folium(m, height=420, use_container_width=True)
    with info_col:
        time_label = f"{display_hour}:{display_minute:02d} (simulated)" if use_sim else now.strftime('%H:%M')
        st.markdown(f"#### Required Intensity — {time_label}")
        st.caption("AI + Camera + Time-of-day + Events")
        if not lights_df.empty:
            for i in range(0, len(lights_df), 2):
                cols = st.columns(2)
                for j, idx in enumerate(range(i, min(i+2, len(lights_df)))):
                    l = lights_df.iloc[idx]
                    lid = l['light_id']
                    iv = light_intensities.get(lid, 0)
                    ov = st.session_state["overrides"].get(lid)
                    with cols[j]:
                        ov_html = f"<div class='override-active'>OVERRIDE: {ov['mode']}</div>" if ov else ""
                        st.markdown(f"""
                        <div class='intensity-card'>
                            <div class='node-badge'>{lid}</div>
                            <div class='big-intensity'>{iv:.0f}%</div>
                            <div class='big-label'>{l['location_name'][:22]}</div>
                            {ov_html}
                        </div>""", unsafe_allow_html=True)

# =========================================================
# TAB 2: LIVE CAMERA STREAM
# =========================================================
with tab2:
    st.markdown("#### Live Traffic Camera Surveillance")
    sel_light = st.selectbox("Monitoring Node", lights_df['light_id'].tolist() if not lights_df.empty else [], key="cam_sel")
    base_i = light_intensities.get(sel_light, 0) if sel_light else 0

    _cam_hour = display_hour
    _is_night = (_cam_hour >= 18 or _cam_hour < 6)
    _cam_poor_vis = False
    if not wx_df.empty:
        try:
            _cam_poor_vis = bool(wx_df.iloc[0]['is_poor_visibility'])
        except:
            pass
    _cam_allowed = _is_night or _cam_poor_vis

    if not _cam_allowed:
        st.warning("Camera is **offline** — daytime with clear visibility. "
                   "Camera activates automatically at night (6 PM - 6 AM) or during poor weather conditions.")
        if st.session_state.get("streaming"):
            st.session_state["streaming"] = False
            worker = st.session_state.get("cam_worker")
            if worker is not None:
                worker.stop()
            st.session_state["cam_worker"] = None

    if "streaming" not in st.session_state:
        st.session_state["streaming"] = False
    if "last_cam_results" not in st.session_state:
        st.session_state["last_cam_results"] = {"trs": 0, "boost": 0, "final": 0, "counts": {}}
    if "cam_worker" not in st.session_state:
        st.session_state["cam_worker"] = None
    if "cam_last_db_write" not in st.session_state:
        st.session_state["cam_last_db_write"] = 0.0

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("Start Live Feed", use_container_width=True, type="primary", disabled=not _cam_allowed):
            st.session_state["streaming"] = True
            worker = st.session_state.get("cam_worker")
            if worker is None or not worker.is_running():
                worker = CameraWorker(camera_index=0, width=640, height=480, infer_every=4)
                worker.start()
                st.session_state["cam_worker"] = worker
    with col_stop:
        if st.button("Stop Feed", use_container_width=True):
            st.session_state["streaming"] = False
            worker = st.session_state.get("cam_worker")
            if worker is not None:
                worker.stop()
            st.session_state["cam_worker"] = None

    @st.fragment(run_every=1.0 if st.session_state["streaming"] else None)
    def camera_feed():
        cam_col, res_col = st.columns([1, 1])

        if not st.session_state["streaming"]:
            with cam_col:
                st.info("Press **Start Live Feed** to begin surveillance.")
            with res_col:
                st.markdown(f"""
                <div class='cam-result-card'>
                    <div class='big-label'>Required Intensity</div>
                    <div class='cam-big'>{base_i:.0f}%</div>
                    <div style='color:rgba(255,255,255,0.35); margin-top:10px; font-size:0.85rem;'>Based on AI + time-of-day</div>
                </div>""", unsafe_allow_html=True)
            return

        _frag_hour = display_hour
        _frag_night = (_frag_hour >= 18 or _frag_hour < 6)
        _frag_poor_vis = False
        if not wx_df.empty:
            try:
                _frag_poor_vis = bool(wx_df.iloc[0]['is_poor_visibility'])
            except:
                pass
        if not (_frag_night or _frag_poor_vis):
            st.session_state["streaming"] = False
            worker = st.session_state.get("cam_worker")
            if worker is not None:
                worker.stop()
                st.session_state["cam_worker"] = None
            with cam_col:
                st.info("Camera offline — clear daytime conditions.")
            return

        worker = st.session_state.get("cam_worker")
        if worker is None or not worker.is_running():
            worker = CameraWorker(camera_index=0, width=640, height=480, infer_every=4)
            worker.start()
            st.session_state["cam_worker"] = worker

        snap = worker.snapshot()
        if snap["error"]:
            with cam_col:
                st.error(snap['error'])
            return

        frame_rgb = snap["frame_rgb"]
        trs = float(snap["trs"])
        boost = float(snap["boost"])
        final_i = min(100.0, base_i + boost)
        det_counts = snap["counts"]

        with cam_col:
            if frame_rgb is not None:
                st.image(frame_rgb, caption=f"LIVE — {sel_light}", use_container_width=True)
            else:
                st.info("Initializing camera stream...")

        with res_col:
            st.markdown(f"""
            <div class='cam-result-card'>
                <div class='big-label'>Required Intensity</div>
                <div class='cam-big'>{final_i:.0f}%</div>
                <div style='margin-top:14px; display:flex; justify-content:center; gap:10px;'>
                    <span class='trs-pill'>TRS: {trs:.1f}</span>
                    <span class='trs-pill'>Boost: +{boost:.0f}%</span>
                </div>
                <div style='color:rgba(255,255,255,0.35); margin-top:10px; font-size:0.85rem;'>
                    Base: {base_i:.0f}% + Camera: {boost:.0f}%
                </div>
            </div>""", unsafe_allow_html=True)

            obj_md = ""
            if det_counts:
                for obj, cnt in det_counts.items():
                    w = [v for k, v in COCO_WEIGHTS.items() if COCO_LABELS.get(k) == obj]
                    wt = w[0] if w else 1.0
                    obj_md += f"- **{obj}** x {cnt} (w={wt}) = `+{cnt * wt:.1f}`"
            else:
                obj_md = "*No traffic objects in frame*"
            st.markdown(f"**Detected Objects**0{obj_md}")

        now_ts = time.time()
        last_write = st.session_state.get("cam_last_db_write", 0.0)
        if now_ts - last_write >= 5.0:
            insert_db(
                "INSERT INTO detection_events (minute_ts, light_id, trs_avg, trs_max, intensity_boost, final_intensity) VALUES (?,?,?,?,?,?)",
                (datetime.datetime.now(), sel_light, trs, trs, boost, final_i),
            )
            st.session_state["cam_last_db_write"] = now_ts

    camera_feed()

    # Override + Forecast
    st.divider()
    ov1, ov2 = st.columns(2)
    with ov1:
        st.markdown("#### Manual Override")
        mode = st.radio("Mode", ["AUTO", "MANUAL", "OFF"], horizontal=True, key="ov_m")
        cval = st.slider("Intensity %", 0, 100, int(base_i), key="ov_s") if mode == "MANUAL" else 0
        if st.button("Apply Override", key="ov_b", use_container_width=True, type="primary"):
            fv = 0 if mode == "OFF" else (cval if mode == "MANUAL" else -1)
            insert_db("INSERT INTO override_events (ts, light_id, user, previous_value, new_value, override_type) VALUES (?,?,?,?,?,?)",
                      (now, sel_light, st.session_state["username"], base_i, fv, mode))
            if mode == "MANUAL":
                st.session_state["overrides"][sel_light] = {"mode": "MANUAL", "value": cval}
                st.success(f"Override applied: MANUAL at {cval}%")
            elif mode == "OFF":
                st.session_state["overrides"][sel_light] = {"mode": "OFF", "value": 0}
                st.success("Override applied: Light OFF")
            else:
                st.session_state["overrides"].pop(sel_light, None)
                st.success("Override released — reverting to AI predictive intensity")
            st.rerun()

        ov_state = st.session_state["overrides"].get(sel_light)
        if ov_state:
            st.warning(f"Active override: **{ov_state['mode']}** at {ov_state['value']}%  Switch to AUTO to release.")
        else:
            st.caption("Mode: AUTO — Using AI prediction + time curve")

    with ov2:
        st.markdown("#### 24-Hour Forecast")
        forecast_times = []
        forecast_vals = []
        for h_offset in range(24):
            tgt_h = (display_hour + h_offset) % 24
            tgt_d = display_date + datetime.timedelta(days=(display_hour + h_offset)//24)
            forecast_times.append(f"{tgt_h:02d}:00")
            forecast_vals.append(compute_base_intensity(sel_light, tgt_h, 0, sim_date=tgt_d, city_code=city))

        fig = go.Figure(go.Scatter(x=forecast_times, y=forecast_vals,
                                   mode='lines+markers',
                                   line=dict(color='#00e5ff', width=2.5, shape='spline'),
                                   marker=dict(size=6, color='#00e5ff', line=dict(width=1, color='white')),
                                   fill='tozeroy',
                                   fillcolor='rgba(0,229,255,0.07)'))
        fig.update_layout(
            height=260,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(22,27,34,0.8)',
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(range=[0, 100], title="Intensity %", gridcolor='rgba(48,54,61,0.5)'),
            xaxis=dict(title="Hour", gridcolor='rgba(48,54,61,0.5)')
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 3: IMAGE ANALYSIS
# =========================================================
with tab3:
    st.markdown("#### Light Intensity & Danger Level Analysis")
    st.caption("Upload an image to analyze light intensity and assess its danger level for light pollution.")

    uploaded_img = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="img_upload")

    if uploaded_img is not None:
        file_bytes = np.frombuffer(uploaded_img.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            avg_brightness = float(np.mean(gray))
            max_brightness = float(np.max(gray))
            brightness_std = float(np.std(gray))
            intensity_pct = (avg_brightness / 255.0) * 100.0
            bright_pixels = float(np.sum(gray > 200)) / gray.size * 100.0

            if intensity_pct > 75 or bright_pixels > 50:
                danger_level, danger_color = "CRITICAL", "#f44336"
                danger_desc = "Extreme light intensity. Causes severe light pollution, disrupts wildlife and human circadian rhythms. Immediate dimming required."
            elif intensity_pct > 55 or bright_pixels > 30:
                danger_level, danger_color = "HIGH", "#ff9800"
                danger_desc = "High light intensity. Significant skyglow contribution. Reduce by 30-50% during off-peak hours."
            elif intensity_pct > 35 or bright_pixels > 15:
                danger_level, danger_color = "MODERATE", "#ffeb3b"
                danger_desc = "Moderate levels. Acceptable for high-traffic but should reduce late night."
            else:
                danger_level, danger_color = "LOW", "#4CAF50"
                danger_desc = "Environmentally friendly. Suitable for deep night / low-traffic."

            img_col, res_col = st.columns([1, 1])
            with img_col:
                st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
                heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), caption="Brightness Heatmap", use_container_width=True)

            with res_col:
                st.markdown(f"""
                <div class='cam-result-card'>
                    <div class='big-label'>Detected Light Intensity</div>
                    <div class='cam-big'>{intensity_pct:.1f}%</div>
                    <div style='margin-top:14px;'>
                        <span style='background:{danger_color};padding:6px 20px;border-radius:20px;color:white;font-weight:bold;font-size:1rem;
                                     letter-spacing:1px;'>
                            {danger_level}
                        </span>
                    </div>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"**Assessment:** {danger_desc}")
                st.markdown("---")
                m1, m2 = st.columns(2)
                m1.metric("Avg Brightness", f"{avg_brightness:.1f} / 255")
                m2.metric("Max Brightness", f"{max_brightness:.0f} / 255")
                m1.metric("Brightness StdDev", f"{brightness_std:.1f}")
                m2.metric("Bright Pixel %", f"{bright_pixels:.1f}%")

                suggested = max(20, intensity_pct * 0.5) if intensity_pct > 60 else (intensity_pct * 0.7 if intensity_pct > 40 else intensity_pct)
                st.metric("Suggested Optimal Intensity", f"{suggested:.0f}%",
                          delta=f"{suggested - intensity_pct:.0f}% from detected", delta_color="normal")
        else:
            st.error("Could not decode the uploaded image.")

# =========================================================
# TAB 4: ANALYTICS
# =========================================================
with tab4:
    st.markdown("#### Energy & Environmental Impact")
    total = len(lights_df)
    avg_i = sum(light_intensities.values()) / max(total, 1)
    power_kw = total * (avg_i / 100) * 0.12
    saved_kw = total * ((100 - avg_i) / 100) * 0.12
    co2 = saved_kw * 12 * 0.82

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#00e5ff;">{avg_i:.1f}%</div>
            <div class="metric-label">Avg Intensity</div>
        </div>""", unsafe_allow_html=True)
    with e2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#ff6b6b;">{power_kw:.2f} kW</div>
            <div class="metric-label">Current Draw</div>
        </div>""", unsafe_allow_html=True)
    with e3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#6bcb77;">{saved_kw:.2f} kW</div>
            <div class="metric-label">Energy Saved</div>
        </div>""", unsafe_allow_html=True)
    with e4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#ffd93d;">{co2:.1f} kg</div>
            <div class="metric-label">CO2 Saved (12h)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    st.markdown("#### AI Model Performance")
    m_df = query_db("SELECT trained_at, model_type, mae, rmse, r2, custom_accuracy FROM model_metrics")
    if not m_df.empty:
        m_df.columns = ['Trained At', 'Model', 'MAE', 'RMSE', 'R2', 'Accuracy %']
        st.dataframe(m_df, use_container_width=True, hide_index=True)
    else:
        st.info("No model metrics yet.")

# =========================================================
# TAB 5: DATA & LOGS
# =========================================================
with tab5:
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("#### Civic Calendar")
        cal = query_db("SELECT id, event_date, event_name, multiplier, city_code FROM celebration_calendar")
        if not cal.empty:
            display_cal = cal.copy()
            display_cal.columns = ['ID', 'Date', 'Event', 'Intensity Multiplier', 'City']
            st.dataframe(display_cal, use_container_width=True, hide_index=True)

            # Edit Event
            st.markdown("##### Edit Event")
            edit_options = {f"{r['id']} — {r['event_name']} ({r['event_date']})": r['id'] for _, r in cal.iterrows()}
            edit_sel = st.selectbox("Select event to edit", list(edit_options.keys()), key="edit_ev_sel")
            sel_id = edit_options[edit_sel]
            sel_row = cal[cal['id'] == sel_id].iloc[0]
            with st.form("edit_event_form"):
                ed_c1, ed_c2 = st.columns(2)
                with ed_c1:
                    ed_name = st.text_input("Event Name", value=sel_row['event_name'])
                    ed_date = st.date_input("Event Date", value=datetime.date.fromisoformat(sel_row['event_date']))
                with ed_c2:
                    ed_mult = st.slider("Intensity Multiplier", 1.0, 2.0, float(sel_row['multiplier']), 0.1, key="ed_mult")
                    ed_city = st.selectbox("Applies To", [city] + ["ALL"],
                                           index=([city] + ["ALL"]).index(sel_row['city_code']) if sel_row['city_code'] in [city, "ALL"] else 0,
                                           key="ed_city")
                ed_submit = st.form_submit_button("Save Changes", use_container_width=True)
                if ed_submit:
                    insert_db("UPDATE celebration_calendar SET event_date=?, event_name=?, multiplier=?, city_code=? WHERE id=?",
                              (str(ed_date), ed_name, ed_mult, ed_city, sel_id))
                    st.success(f"Updated event #{sel_id}.")
                    st.rerun()

            # Delete Event
            st.markdown("##### Delete Event")
            del_sel = st.selectbox("Select event to delete", list(edit_options.keys()), key="del_ev_sel")
            del_id = edit_options[del_sel]
            if st.button("Delete Selected Event", key="del_ev_btn", type="secondary"):
                insert_db("DELETE FROM celebration_calendar WHERE id = ?", (del_id,))
                st.success(f"Deleted event #{del_id}.")
                st.rerun()
        else:
            st.info("No events recorded.")

    st.divider()
    st.markdown("#### Add Event / Concert / Celebration")
    st.caption("Adding an event boosts street light intensity on that date via the multiplier.")
    with st.form("add_event_form", clear_on_submit=True):
        ev_c1, ev_c2 = st.columns(2)
        with ev_c1:
            ev_name = st.text_input("Event Name", placeholder="e.g. Nishagandhi Concert")
            ev_date = st.date_input("Event Date", value=datetime.date.today())
        with ev_c2:
            ev_mult = st.slider("Intensity Multiplier", 1.0, 2.0, 1.3, 0.1,
                                help="1.0 = no change, 1.5 = 50% brighter, 2.0 = full brightness")
            ev_city = st.selectbox("Applies To", [city] + ["ALL"], key="ev_city")
        ev_note = st.text_input("Area / Notes", placeholder="e.g. Kanakakkunnu Palace area")
        submitted = st.form_submit_button("Add Event", use_container_width=True, type="primary")
        if submitted and ev_name:
            full_name = f"{ev_name} ({ev_note})" if ev_note else ev_name
            insert_db("INSERT INTO celebration_calendar (event_date, event_name, multiplier, city_code) VALUES (?,?,?,?)",
                      (str(ev_date), full_name, ev_mult, ev_city))
            st.success(f"Added '{full_name}' on {ev_date} with {ev_mult}x intensity boost.")
            st.rerun()
    with d2:
        st.markdown("#### VIIRS Satellite Data")
        v = query_db("SELECT date, light_id, raw_radiance, skyglow_index, quality_flag FROM viirs_data")
        if not v.empty:
            v.columns = ['Date', 'Node', 'Radiance', 'Skyglow Index', 'Quality']
            st.dataframe(v, use_container_width=True, hide_index=True)
        else:
            st.info("No satellite data available.")

    st.divider()
    st.markdown("#### System Logs")
    logs = query_db("SELECT ts, level, component, message FROM system_log ORDER BY ts DESC LIMIT 50")
    if not logs.empty:
        logs.columns = ['Timestamp', 'Level', 'Module', 'Message']
        st.dataframe(logs, use_container_width=True, hide_index=True)
    else:
        st.info("No logs recorded yet.")

    st.divider()
    st.markdown("#### Recent Camera Detections")
    det = query_db("SELECT minute_ts, light_id, trs_avg, intensity_boost, final_intensity FROM detection_events ORDER BY minute_ts DESC LIMIT 20")
    if not det.empty:
        det.columns = ['Timestamp', 'Node', 'TRS', 'Boost %', 'Final Intensity %']
        st.dataframe(det, use_container_width=True, hide_index=True)
    else:
        st.info("No detections yet — use Camera Surveillance tab to start capturing.")

