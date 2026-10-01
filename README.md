<div align="center">

# 💡 LumiCity Kerala
### *Autonomous Smart City Street Light Optimization & Ecological Light Pollution Mitigation Engine*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://ultralytics.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Gradient%20Boosting-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![SQLite](https://img.shields.io/badge/SQLite-WAL%20Mode-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![NASA EarthData](https://img.shields.io/badge/NASA%20VIIRS-VNP46A1%20DNB-0B3D91?style=for-the-badge&logo=nasa&logoColor=white)](https://earthdata.nasa.gov/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<br/>

<p align="center">
  <b>A state-of-the-art municipal IoT and AI platform engineered for 8 major Kerala urban centers.</b><br/>
  <i>Replaces primitive mechanical timers with continuous machine learning, orbital NASA satellite radiance telemetry, real-time edge computer vision, and dynamic weather safety overrides.</i>
</p>

[Explore Documentation](DOCUMENTATION.md) • [View Architecture](#-system-architecture) • [Getting Started](#-installation--quick-start) • [Default Credentials](#-municipal-access--credentials)

</div>

---

## 📑 Table of Contents
- [Urban Crisis & The LumiCity Paradigm](#-urban-crisis--the-lumicity-paradigm)
- [Key Architectural Highlights](#-key-architectural-highlights)
- [System Architecture & Dataflow](#-system-architecture--dataflow)
- [Mathematical Formulation & Physics Engine](#-mathematical-formulation--physics-engine)
- [Machine Learning & Computer Vision Stack](#-machine-learning--computer-vision-stack)
- [Screen-by-Screen Dashboard Tour](#-screen-by-screen-dashboard-tour)
- [Municipal Access & Credentials](#-municipal-access--credentials)
- [Project Directory Manifest](#-project-directory-manifest)
- [Installation & Quick Start](#-installation--quick-start)
- [Hardware & Edge Appliance Notes](#-hardware--edge-appliance-notes)
- [Research Publications & Documentation](#-research-publications--documentation)
- [License & Contributing](#-license--contributing)

---

## 🌌 Urban Crisis & The LumiCity Paradigm

### The Dual Crisis of Modern Municipalities
Traditional municipal street lighting networks operate on **binary mechanical timers** (or basic ambient photocells): street fixtures ignite at 100% full power at 6:00 PM and burn at full intensity until 6:00 AM, regardless of road demand, weather conditions, or civic events.

```
Conventional Timer:   [ 100% Full Blast Illumination — 12 Hours Uninterrupted ]
                      🚨 High Energy Burn | 🚨 Massive Skyglow | 🚨 Glare & Ecological Harm

LumiCity Optimization: [ Dynamic Demand-Responsive Curve: 0% - 100% Adaptive ]
                      🌱 40–70% Energy Saved | 🛡️ Instant Safety Boosts | 🌌 Preserved Night Sky
```

1. **Severe Skyglow (Light Pollution):** High-mast LED photons scatter across atmospheric aerosols and tropical humidity (Rayleigh & Mie scattering), creating luminous artificial domes that eliminate the night sky and disorient nocturnal avian wildlife and sea turtle nesting corridors along Kerala's coastline.
2. **Melatonin Disruption & Light Trespass:** Unfocused photon leakage streams into residential bedrooms, suppressing human melatonin synthesis and inducing circadian disruption.
3. **Disability Glare:** An excessively intense light fixture at 3:00 AM creates pupil constriction against the pitch-black peripheral roadway, reducing emergency reaction time for motorists.
4. **Severe Electrical Budget Waste:** Millions of kilowatt-hours (kWh) and associated carbon footprints are exhausted illuminating deserted highways between midnight and 5:00 AM.

### The LumiCity Solution
LumiCity treats street lighting as a **continuous, real-time mathematical optimization problem**:

$$\text{Intensity}_{\text{req}} = f(\text{Time}, \text{Radiance}_{\text{VIIRS}}, \text{Visibility}, \text{TrafficRisk}_{\text{YOLO}}, \text{CelebrationMultiplier}, \text{AmbientLux})$$

The platform continuously seeks to **minimize illumination wattage** to protect the biosphere and conserve energy, while enforcing strict **hard-bounded safety guarantees**:
* If atmospheric visibility drops below 3,000 meters (e.g. dense monsoon deluge or heavy winter fog), the system instantaneously mandates an immovable safety floor of **$\ge 60\%$ intensity**, even during daytime.
* If late-night edge cameras detect approaching vehicles or pedestrians, lighting surges dynamically by **$+10\%$ to $+40\%$** before smoothly receding.

---

## ✨ Key Architectural Highlights

* 🛰️ **NASA VIIRS Satellite Telemetry:** Direct ingestion of scientific-grade orbital radiance measurements from Suomi NPP & NOAA-20 Day/Night Band (VNP46A1) to calibrate regional baseline light emissions.
* 🌦️ **Meteorological API Integration:** Real-time ingestion from Open-Meteo for hourly cloud cover, humidity, and atmospheric visibility across all 8 Kerala municipal zones.
* 🧠 **Regularized Gradient Boosting Regressor (GBR):** Fast, explainable decision-tree ensemble ($R^2 \approx 0.92$) featuring deliberate 6% Gaussian noise injection to eliminate geometric overfitting.
* 👁️ **YOLOv8 Edge Computer Vision:** DirectShow OpenCV stream running Ultralytics YOLOv8 nano in a decoupled background daemon to track vehicular/pedestrian volume and score traffic hazard in real time.
* 🚗 **Kinetic Energy Weighted Traffic Risk Score (TRS):** Objects are mathematically weighted by physical momentum ($K = \frac{1}{2}mv^2$): trucks and buses provide higher intensity boosts than lightweight cars or bicycles.
* 📈 **Minute-Level Smooth Spline Interpolation:** 21-point anchor curve with continuous linear interpolation across 24 hours to prevent sudden step-function jumps that cause pupil shock.
* 🔄 **Asynchronous Multi-Threaded Daemon Architecture:** Dual background workers (`CameraWorker` and `DataRefreshWorker`) prevent UI thread freezing and maintain continuous 5-hour weather and 24-hour prediction refreshes.
* 💎 **Ultra-Modern Glassmorphism UI:** Cyberpunk-inspired dark aesthetic (`#0a0e17` to `#111827`), translucent cards, live JavaScript IST clock, breathing SVG status dots, and partial `@st.fragment` re-rendering.
* 🗺️ **Interactive Folium GIS Topography:** High-performance CartoDB Dark Matter map plotting 70 smart fixtures color-coded by real-time operational status (Green = Optimized, Orange = Medium, Red = High, Purple = Manual Override).
* 🔬 **Light Pollution Image Profiler:** Computer vision upload diagnostic tool converting citizen street photos into grayscale brightness matrices, JET colormaps, and ecological danger classifications.
* 📊 **Automated Carbon Accounting:** Real-time calculation of kilowatts drawn, electrical power saved, and carbon offset ($CO_2$ kilograms) over 12-hour cycles.

---

## 🏛️ System Architecture

```mermaid
flowchart TB
    subgraph External_APIs [External Satellite & Atmospheric APIs]
        NASA["🛰️ NASA LAADS DAAC<br/>VIIRS VNP46A1 Day/Night Band"]
        Meteo["🌦️ Open-Meteo REST API<br/>Cloudcover & Visibility"]
    end

    subgraph Offline_ML [Offline Data Engineering & AI Training Pipeline]
        Jupyter["📓 Jupyter Notebook (LumiCity_AI.ipynb)"]
        NASA -->|HDF5 Granule Radiance| Jupyter
        Meteo -->|Hourly Weather Metrics| Jupyter
        Jupyter -->|Feature Engineering (10 Features)| Train["GBR Regressor Training<br/>(200 Trees, 6% Gaussian Noise)"]
        Train --> ModelExport["💾 intensity_model.pkl<br/>💾 intensity_scaler.pkl"]
    end

    subgraph Relational_Core [Localized Edge Persistence - SQLite WAL Mode]
        DB[("🗄️ kerala_smart_lights.db<br/>• street_lights (70 Nodes)<br/>• city_zones (8 Municipalities)<br/>• predictions (1,680 Rows)<br/>• weather_data & viirs_data<br/>• celebration_calendar<br/>• detection_events & override_events<br/>• model_metrics & system_log")]
    end

    subgraph Backend_Daemons [Asynchronous Background Threads]
        DRW["🔄 DataRefreshWorker Daemon<br/>• 5h Weather Cycle<br/>• 24h Model Prediction Cycle"]
        CW["🎥 CameraWorker Daemon<br/>• OpenCV VideoCapture (30fps)<br/>• YOLOv8n Tensor Inference<br/>• Thread-Safe Mutex (threading.Lock)<br/>• Throttled DB Writes (5.0s)"]
    end

    subgraph Presentation [Streamlit Interactive Web Portal - app.py]
        UI["🖥️ Modern Glassmorphism Dashboard"]
        Tab1["🗺️ Tab 1: GIS Map & Intensity Cards"]
        Tab2["📹 Tab 2: Live YOLO Stream & Override"]
        Tab3["🔬 Tab 3: Light Pollution Profiler"]
        Tab4["📈 Tab 4: Energy & CO2 Analytics"]
        Tab5["📋 Tab 5: Civic Calendar & Logs"]
    end

    External_APIs <-->|REST & Streaming| DRW
    ModelExport -->|Load at Startup| DRW
    ModelExport -->|Load at Startup| UI
    DRW <-->|Read / Write| DB
    CW <-->|Throttled TRS Writes| DB
    DB <-->|Direct SQL Bridges| UI
    CW -.->|Real-time Snapshot| Tab2
    UI --- Tab1 & Tab2 & Tab3 & Tab4 & Tab5
```

---

## 📐 Mathematical Formulation & Physics Engine

### 1. Commuter Traffic Proxy Formulation
To capture urban commute surges without expensive, privacy-invasive live GPS tracking, LumiCity employs overlapping harmonic sinusoids:

$$\text{TrafficFactor}(h) = \max\left(0.1, \sin\left(\frac{\pi(h - 4)}{12}\right) + 0.5 \cos\left(\frac{\pi h}{6}\right)\right)$$

* **Primary Sine Wave:** Phase-shifted by 4 hours, anchoring the morning commute at 4:00 AM, peaking near mid-day solar noon, and tapering off after sunset.
* **Secondary Cosine Harmonic:** Injects the evening "second rush hour" bump between 17:00 and 19:30.
* **Zero-Bound Floor ($0.1$):** Guarantees baseline illumination for emergency vehicles, night-shift transit, and police patrols.

### 2. Kinetic-Weighted Traffic Risk Score (TRS)
When camera surveillance is active, incoming OpenCV frames are passed through the YOLOv8 nano neural tensor. Detected entities are weighted based on Newtonian momentum ($K = \frac{1}{2}mv^2$ and required braking distance):

$$\text{TRS} = \sum_{i \in \text{Objects}} W_{\text{COCO}}[\text{class}_i]$$

| COCO Class ID | Entity Label | Weight ($W$) | Kinetic Rationale |
|:---:|:---|:---:|:---|
| `0` | **Person** | `1.0` | Vulnerable pedestrian; maximum sensitivity |
| `1` | **Bicycle** | `1.2` | Slow-moving non-motorized commuter |
| `3` | **Motorcycle** | `1.5` | Fast-moving, low-mass motorized vehicle |
| `2` | **Car** | `2.0` | Standard passenger automobile |
| `5` | **Bus** | `3.5` | High-mass mass transit; long braking distance |
| `7` | **Truck** | `3.5` | Heavy commercial freight; maximum momentum |

$$\text{Boost}(\text{TRS}) = \begin{cases} 
+40\% & \text{if } \text{TRS} \ge 15.0 \quad \text{(Traffic Congestion)} \\
+30\% & \text{if } \text{TRS} \ge 10.0 \quad \text{(Dense Vehicular Flow)} \\
+20\% & \text{if } \text{TRS} \ge 5.0 \quad \text{(Moderate Movement)} \\
+10\% & \text{if } \text{TRS} \ge 2.0 \quad \text{(Sporadic Traffic)} \\
0\% & \text{otherwise} \quad \text{(Empty Road; Revert to AI)}
\end{cases}$$

### 3. Minute-Level Continuous Spline Blending
To prevent sudden step-function jumps at the top of the hour, `compute_base_intensity()` linearly interpolates between adjacent temporal anchors:

$$\text{Base}(t) = v_0 + (v_1 - v_0) \times \left(\frac{t - t_0}{t_1 - t_0}\right)$$

During nighttime hours, the baseline is fused with the Machine Learning prediction:
$$\text{FinalIntensity} = \min\left(100, \left(0.7 \times \text{Prediction}_{\text{GBR}} + 0.3 \times \text{Base}(t)\right) \times \text{Multiplier}_{\text{Festival}} + \text{Boost}_{\text{Camera}}\right)$$

### 4. Skyglow Ecological Metric & Carbon Accounting
* **Real-Time Skyglow Proxy:**
  $$\text{Skyglow Index} = \overline{\text{Radiance}}_{\text{VIIRS}} \times \left(\frac{\overline{\text{Intensity}}}{100}\right)$$
  *(Green $< 0.8$ = Safe, Orange $0.8–1.5$ = Moderate, Red $> 1.5$ = Severe Ecological Degradation)*
* **Power Draw & Carbon Savings:**
  $$\text{Power}_{\text{Current}} = N_{\text{nodes}} \times \left(\frac{\overline{\text{Intensity}}}{100}\right) \times 0.12\text{ kW}$$
  $$\text{Power}_{\text{Saved}} = N_{\text{nodes}} \times \left(\frac{100 - \overline{\text{Intensity}}}{100}\right) \times 0.12\text{ kW}$$
  $$\text{CO}_2\text{ Offset (12h)} = \text{Power}_{\text{Saved}} \times 12\text{ hours} \times 0.82\text{ kg CO}_2/\text{kWh}$$

---

## 🛠️ Machine Learning & Computer Vision Stack

```
Input Features (10 Dimensions)
├── hour (0-23)
├── weekday (0-6)
├── is_weekend (0 or 1)
├── month (1-12)
├── is_poor_visibility (0 or 1, threshold: 3000m)
├── viirs_radiance (NASA DNB satellite ground truth)
├── celebration_multiplier (Civic festival calendar: 1.0 - 2.0)
├── traffic_proxy (Sinusoidal commuter proxy)
├── latitude (Fixture GPS)
└── longitude (Fixture GPS)
        │
        ▼
StandardScaler Normalization (intensity_scaler.pkl)
        │
        ▼
GradientBoostingRegressor (intensity_model.pkl)
├── n_estimators = 200 trees
├── learning_rate = 0.1
├── max_depth = 4
└── Regularization: Gaussian Noise N(0, 6)
        │
        ▼
Predicted Intensity (0% - 100%)
[MAE: 3.2% | RMSE: 4.6% | R²: 0.92 | Accuracy within ±5%: 89.4%]
```

---

## 🖥️ Screen-by-Screen Dashboard Tour

### 1. Cyberpunk Glassmorphism Authentication
Secure SHA-256 login with municipality routing. Superusers (`admin`) gain global authority to toggle across any city at runtime.

### 2. Tab 1: City Map & Street Light Intensity Grid
* **Folium GIS Map:** Renders city boundaries and fixture coordinates on CartoDB Dark Matter tiles. Circular node markers glow dynamically based on current intensity:
  * 🟢 **Green (`#4CAF50`):** Eco-optimized low power ($< 40\%$)
  * 🟠 **Orange (`#ff9800`):** Mid-tier active street ($40\% - 70\%$)
  * 🔴 **Red (`#f44336`):** High intensity draw ($> 70\%$)
  * 🟣 **Purple (`#9c27b0`):** Active municipal operator override
* **Interactive Node Cards:** Two-column glass card layout displaying real-time dimming percentages and fixture identifiers (`NODE-TVM-A01`, `NODE-KCH-B04`, etc.).

### 3. Tab 2: Live Camera Surveillance & Manual Override
* **Real-Time YOLOv8 Stream:** Decoupled webcam inference displaying bounding boxes, tracked class counts, and instantaneous TRS metrics.
* **Smart Activation:** Automatically conserves system compute during clear daylight; activates automatically at night (6:00 PM – 6:00 AM) or during low-visibility weather.
* **Direct Municipal Override:** Allows authorized operators to force lights to `AUTO` (AI baseline), `MANUAL` (custom slider 0–100%), or `OFF` with full audit logging.
* **24-Hour Predictive Horizon:** Interactive Plotly spline graph forecasting fixture power over the upcoming day.

### 4. Tab 3: Light Pollution Profiler (Diagnostic Inspector)
* Upload high-resolution night sky or roadway photography (`PNG`, `JPG`).
* Extracts average luminance, maximum pixel saturation, variance, and bright pixel percentage ($>200$).
* Generates an OpenCV **JET Colormap Heatmap** revealing localized photon leakage.
* Produces scientific severity ratings (**CRITICAL**, **HIGH**, **MODERATE**, **LOW**) along with mathematically recommended dimming adjustments.

### 5. Tab 4: Energy & Environmental Impact Analytics
* High-visibility metric cards tracking active municipal power draw, kilowatts saved, and total 12-hour carbon offset in kilograms of $CO_2$.
* ML model performance leaderboard reporting Mean Absolute Error, Root Mean Squared Error, $R^2$, and Custom Safety Accuracy Thresholds (CSAT).

### 6. Tab 5: Civic Calendar & Audit Ledger
* **Festival & Celebration Management:** Add, edit, or delete civic events (e.g. Onam, Vishu, New Year, concerts) with custom brightness multipliers.
* **NASA VIIRS Telemetry Table:** Inspect ground-truth radiance values downloaded from satellite passes.
* **Multi-Tier Audit Logs:** Real-time log monitoring operator overrides, YOLO detections, and background daemon cycles.

---

## 🔐 Municipal Access & Credentials

The system provides pre-configured credentials hashed with SHA-256 in `credentials.json`:

| Username | Default Password | Administrative Authority | Assigned Municipality |
|:---|:---|:---|:---|
| **`admin`** | `admin123` | **Superuser (All Municipalities)** | Global Kerala Switcher |
| **`tvm_admin`** | `tvm123` | Regional Engineer | Thiruvananthapuram (`TVM`) |
| **`kch_admin`** | `kochi123` | Regional Engineer | Kochi (`KCH`) |
| **`kzd_admin`** | `kzd123` | Regional Engineer | Kozhikode (`KZD`) |
| **`tcr_admin`** | `tcr123` | Regional Engineer | Thrissur (`TCR`) |
| **`klm_admin`** | `klm123` | Regional Engineer | Kollam (`KLM`) |
| **`knr_admin`** | `knr123` | Regional Engineer | Kannur (`KNR`) |
| **`alp_admin`** | `alp123` | Regional Engineer | Alappuzha (`ALP`) |
| **`pkd_admin`** | `pkd123` | Regional Engineer | Palakkad (`PKD`) |

---

## 📂 Project Directory Manifest

```
LumiCity/
│
├── app.py                              # Core Streamlit application (1,520 lines of code)
├── LumiCity_AI.ipynb                   # Step-by-step ML pipeline & data extraction notebook
├── requirements.txt                    # Production pip package dependencies
├── .gitignore                          # Standard git exclusion rules
├── credentials.json                    # SHA-256 hashed municipal login credentials
├── kerala_smart_lights.db              # Relational SQLite database in WAL mode (10 tables)
├── intensity_model.pkl                 # Serialized scikit-learn GradientBoostingRegressor
├── intensity_scaler.pkl                # Serialized StandardScaler transformation model
├── yolov8n.pt                          # Ultralytics YOLOv8 nano pre-trained weights
│
├── LumiCity_Run_Guide.pdf              # Operational execution and quick-start guide
├── LumiCity_Documentation.pdf          # Full academic research & implementation report
├── LumiCity_Kerala_Full_Documentation.pdf # 58-page comprehensive architectural deep-dive
│
├── DOCUMENTATION.md                    # Exhaustive technical system documentation & status tracker
├── README.md                           # Public repository showcase & instructions
└── viirs_*.h5 (15 files)               # Sample NASA VIIRS Day/Night Band HDF5 granules
```

---

## 🚀 Installation & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/navvv29/LumiCity.git
cd LumiCity
```

### 2. Create and Activate a Virtual Environment
* **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
* **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the Streamlit Portal
```bash
streamlit run app.py
```
After initialization, your browser will automatically open:
```
Local URL: http://localhost:8501
```

### 5. Sign In
Log in with username **`admin`** and password **`admin123`** to access all municipal zones and capabilities.

---

## ⚡ Hardware & Edge Appliance Notes

* **Camera Interface:** `app.py` initializes OpenCV with `cv2.CAP_DSHOW` (DirectShow backend for Windows). If deploying to a Linux-based edge appliance (e.g. Raspberry Pi 5 or NVIDIA Jetson Orin), OpenCV will automatically interface with standard V4L2 drivers (`/dev/video0`).
* **Database Concurrency:** SQLite is configured with **Write-Ahead Logging (`PRAGMA journal_mode=WAL`)** and a 10-second timeout lock to support non-blocking concurrent reads and writes between Streamlit and daemon threads.
* **Offline Resiliency:** If external internet connectivity to NASA or Open-Meteo drops, the background worker smoothly falls back to regional median radiances (`15.4 nW/cm²/sr`) and cached weather records without crashing.

---

## 📚 Research Publications & Documentation

Comprehensive PDF documentation is included directly within the repository for academic reference and architectural audit:
* 📄 **[`LumiCity_Kerala_Full_Documentation.pdf`](LumiCity_Kerala_Full_Documentation.pdf):** 58-page comprehensive architectural, mathematical, and code-level deep-dive.
* 📄 **[`LumiCity_Documentation.pdf`](LumiCity_Documentation.pdf):** Complete project summary report and system findings.
* 📄 **[`LumiCity_Run_Guide.pdf`](LumiCity_Run_Guide.pdf):** Step-by-step setup and operational run guide.
* 📄 **[`DOCUMENTATION.md`](DOCUMENTATION.md):** Markdown technical status ledger and component specifications.

---

## 📄 License & Contributing

This project is licensed under the **MIT License**. Contributions, issues, and feature suggestions are welcome!

<div align="center">
  <sub>Engineered with precision for Kerala Municipal Corporations • Developed by <b>Navaneed P</b></sub>
</div>
