# 👁️ DrowsAlert — Driver Fatigue Detection

A real-time drowsiness and yawning detection system built with **OpenCV**, **dlib**, and **Streamlit**. It analyses facial landmarks to compute Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR), flagging signs of driver fatigue in real time.

---

## Features

- **Webcam snapshot analysis** — capture a frame via browser and get instant EAR/MAR readings
- **Batch image analysis** — upload multiple images, get annotated results with a download button for each
- **Adjustable thresholds** — tune EAR and MAR sensitivity live from the sidebar
- **68-point landmark overlay** — optional mesh drawn over detected faces
- **Multi-face support** — detects and reports on multiple faces in a single frame

---

## How It Works

Detection relies on two geometric ratios computed from dlib's 68-point facial landmark model.

### EAR — Eye Aspect Ratio

```
EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)
```

Six landmark points per eye are used. When the eye is open, EAR stays roughly between 0.25–0.35. When the eye closes, it drops sharply toward zero. If EAR falls below the configured threshold, drowsiness is flagged.

### MAR — Mouth Aspect Ratio

The same approach is applied to the 20 mouth landmark points. A high MAR value indicates a wide-open mouth, which is treated as a yawn event. Unlike EAR, yawning is detected on a single-frame basis.

### Detection Pipeline

1. Convert frame to grayscale
2. dlib HOG face detector finds bounding boxes
3. 68-point shape predictor localises facial geometry
4. Extract eye landmarks (points 36–47) and mouth landmarks (points 48–67)
5. Compute EAR and MAR per detected face
6. Compare against thresholds → trigger alert

---

## Project Structure

```
.
├── app.py                                  # Streamlit deployment
├── drowsiness_detection.py                 # Original CLI script (webcam + dataset modes)
├── shape_predictor_68_face_landmarks.dat   # dlib landmark model (download separately)
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/drowsalert.git
cd drowsalert
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `dlib` requires CMake and a C++ compiler. On Ubuntu: `sudo apt install cmake build-essential`. On Windows, install CMake from cmake.org and use Visual Studio Build Tools.

### 3. Download the landmark model

The `shape_predictor_68_face_landmarks.dat` file is required but not included in this repo due to its size (~100 MB).

Download it from the [dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), extract it, and place it in the project root alongside `app.py`.

```bash
# Linux / macOS
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Running the Original CLI Script

The original `drowsiness_detection.py` supports two modes:

```bash
python drowsiness_detection.py
```

```
A. webcam (enter 1)
B. sample image (enter 2)
```

- **Mode 1** — opens the default webcam, runs detection live, press `ESC` to quit
- **Mode 2** — prompts for a folder path and processes all images (`.jpg`, `.jpeg`, `.png`, `.bmp`) in it

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| EAR Threshold | `0.25` | EAR below this value → drowsy |
| MAR Threshold | `0.75` | MAR above this value → yawning |
| Draw Landmark Mesh | `True` | Overlay 68-point landmarks on the frame |

These can be adjusted live via the sidebar in the Streamlit app, or changed directly in `drowsiness_detection.py` at the top of the file.

---

## Dependencies

```
streamlit>=1.32.0
opencv-python-headless>=4.8.0
dlib>=19.24.0
numpy>=1.24.0
scipy>=1.11.0
Pillow>=10.0.0
```

---

## Landmark Reference

| Points | Region |
|--------|--------|
| 0–16 | Jawline |
| 17–21 | Left eyebrow |
| 22–26 | Right eyebrow |
| 27–35 | Nose |
| 36–41 | Left eye (EAR) |
| 42–47 | Right eye (EAR) |
| 48–67 | Mouth (MAR) |

---

## References

- Soukupová & Čech (2016) — *Real-Time Eye Blink Detection using Facial Landmarks* — the paper that introduced EAR
- [dlib](http://dlib.net/) — face detection and shape prediction
- [iBUG 300-W dataset](https://ibug.doc.ic.ac.uk/resources/300-W/) — used to train the 68-point landmark model

## Authors

| # | Name | Roll No. | GitHub |
|---|------|----------|--------|
| 1 | **Sherry Singh** | 102323042 | [@sherrysingh1410](https://github.com/sherrysingh1410) |
| 2 | **Idhant Mehta** | 102323064 | [@Idhant-Mehta](https://github.com/Idhant-Mehta) |
| 3 | **Sparsh** | 102323080 | [@sparsh0106](https://github.com/sparsh0106) |
