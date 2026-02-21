# Driver Drowsiness Detection System

### UCS532 — Computer Vision (GAMMA) | RAI - 3W1C

> A real-time driver drowsiness detection system built as part of the **UCS532 Computer Vision** course. It monitors the driver's eyes using computer vision techniques and triggers an alert when signs of drowsiness are detected — helping prevent accidents caused by fatigue.

---

## Overview

Drowsy driving is a leading cause of road accidents worldwide. This project leverages **computer vision** and **deep learning** techniques to detect driver drowsiness in real-time and trigger an alert to prevent accidents.

## Features

- **Real-time face & eye detection** using OpenCV and dlib
- **Drowsiness detection** based on Eye Aspect Ratio (EAR)
- **Audio alert system** to wake up the driver
- **Accuracy metrics** and performance evaluation
- **Live webcam feed** with visual indicators

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| OpenCV | Image processing & video capture |
| dlib | Facial landmark detection |
| TensorFlow / Keras | Deep learning model |
| NumPy | Numerical computations |
| Pygame / playsound | Audio alert |

## 📖 How It Works

1. The webcam captures the driver's face in real-time.
2. Facial landmarks are detected using **dlib's 68-point landmark predictor**.
3. The **Eye Aspect Ratio (EAR)** is calculated for both eyes.
4. If the EAR falls below a threshold for a consecutive number of frames, the system triggers a **drowsiness alert**.

## 🔧 Scope of Improvement

- ⚡ **Model Optimization** — Improve model accuracy by training on larger and more diverse datasets across different lighting conditions, ethnicities, and age groups.
- **Low-Light Performance** — Enhance detection reliability in nighttime or low-light driving conditions using infrared (IR) camera support.
- **Reduce False Positives** — Fine-tune EAR thresholds and frame-count parameters to minimize false drowsiness alerts (e.g., during natural blinking).
- **Multi-Feature Detection** — Combine eye-based detection with additional cues like head pose estimation and mouth yawning for more robust results.
- **Cross-Platform Support** — Optimize the system to run efficiently on low-power devices like Raspberry Pi or mobile phones.
- **Latency Reduction** — Reduce processing latency for smoother real-time detection, especially on systems without GPU support.

## Future Updates

| Version | Planned Feature | Status |
|---------|----------------|--------|
| v2.0 | **Yawn Detection** — Detect driver yawning using Mouth Aspect Ratio (MAR) as an additional drowsiness indicator | Planned |
| v2.1 | **Head Pose Estimation** — Monitor head tilting and nodding to detect micro-sleeps | Planned |
| v2.2 | **Mobile App Integration** — Android/iOS app with real-time camera feed and push notifications | Planned |
| v2.3 | **Cloud Dashboard** — Web-based dashboard for fleet managers to monitor driver alertness remotely | Planned |
| v3.0 | **Edge AI Deployment** — Deploy optimized model on Raspberry Pi / NVIDIA Jetson for in-vehicle use | Planned |
| v3.1 | **Voice Alerts & Interaction** — Voice-based warnings and driver interaction for a hands-free experience | Planned |
| v3.2 | **Driver Fatigue Analytics** — Log drowsiness events over time and generate fatigue reports | Planned |

## Authors

| # | Name | Roll No. | GitHub |
|---|------|----------|--------|
| 1 | **Sherry Singh** | 102323042 | [@sherrysingh1410](https://github.com/sherrysingh1410) |
| 2 | **Idhant Mehta** | 102323064 | [@Idhant-Mehta](https://github.com/Idhant-Mehta) |
| 3 | **Sparsh** | 102323080 | [@sparsh0106](https://github.com/sparsh0106) |
| 3 | **Garv Talwar** | 102373005 | [@garvtalwar](https://github.com/garvtalwar) |

