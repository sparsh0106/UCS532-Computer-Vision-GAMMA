# DrowsAlert: Driver Fatigue Detection

DrowsAlert is a computer-vision course project that estimates eye and mouth aspect ratios from facial landmarks to flag possible drowsiness and yawning. It includes a Streamlit interface for image analysis and a separate Python webcam/image script.

## Features

- Analyze webcam captures or uploaded images in the Streamlit app.
- Compute Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
- Adjust detection thresholds and view facial landmarks.

## Run locally

```bash
git clone https://github.com/sparsh0106/UCS532-Computer-Vision-GAMMA.git
cd UCS532-Computer-Vision-GAMMA
python -m pip install -r requirements.txt
```

The dlib 68-point shape predictor is not included in the repository. Download the model from the [dlib model files page](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), extract it, and place `shape_predictor_68_face_landmarks.dat` in the project root.

Start the web interface:

```bash
streamlit run app.py
```

The separate `driver_3_python.py` script contains the original OpenCV/dlib workflow.

## Limitations

This is a student project and a visual alert prototype. It has not been validated for use while driving and must not replace safe driving practices or commercial driver-monitoring systems.
