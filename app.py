import streamlit as st
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from PIL import Image
import io

st.set_page_config(
    page_title="DrowsAlert · Driver Monitoring",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CORE DETECTION LOGIC
# ─────────────────────────────────────────────
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

LEFT_EYE  = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH     = list(range(48, 68))


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading facial landmark model…")
def load_model():
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor


# ─────────────────────────────────────────────
#  FRAME PROCESSING
# ─────────────────────────────────────────────
def process_frame(frame, detector, predictor, ear_thresh, mar_thresh, draw_mesh=True):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    results = []

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame, results

    for i, face in enumerate(faces):
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        leftEye  = shape[LEFT_EYE]
        rightEye = shape[RIGHT_EYE]
        mouth    = shape[MOUTH]

        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        drowsy  = ear < ear_thresh
        yawning = mar > mar_thresh

        results.append({"face": i + 1, "ear": ear, "mar": mar,
                        "drowsy": drowsy, "yawning": yawning})

        if draw_mesh:
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            box_color = (0, 0, 255) if drowsy else (0, 165, 255) if yawning else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR: {mar:.3f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        if drowsy:
            cv2.putText(frame, "DROWSY!", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if yawning:
            cv2.putText(frame, "YAWNING!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

    return frame, results


# ─────────────────────────────────────────────
#  RENDER RESULTS
# ─────────────────────────────────────────────
def render_results(results, ear_thresh, mar_thresh):
    if not results:
        st.warning("No face detected in frame.")
        return

    for r in results:
        if len(results) > 1:
            st.subheader(f"Face {r['face']}")

        col1, col2 = st.columns(2)
        col1.metric("EAR (Eye Aspect Ratio)", f"{r['ear']:.3f}",
                    delta=f"threshold: {ear_thresh:.2f}", delta_color="off")
        col2.metric("MAR (Mouth Aspect Ratio)", f"{r['mar']:.3f}",
                    delta=f"threshold: {mar_thresh:.2f}", delta_color="off")

        if r['drowsy']:
            st.error("😴 DROWSINESS DETECTED — Eyes closed too long")
        if r['yawning']:
            st.warning("🥱 YAWNING DETECTED — Fatigue indicator")
        if not r['drowsy'] and not r['yawning']:
            st.success("✅ Driver appears alert")


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Parameters")
st.sidebar.caption("Tune detection sensitivity")

ear_thresh    = st.sidebar.slider("EAR Threshold (Drowsiness)", 0.10, 0.40, 0.25, 0.01,
                                   help="EAR below this value → drowsy")
mar_thresh    = st.sidebar.slider("MAR Threshold (Yawning)",    0.40, 1.20, 0.75, 0.01,
                                   help="MAR above this value → yawning")
draw_mesh     = st.sidebar.checkbox("Draw 68-Point Landmark Mesh", value=True)

st.sidebar.divider()
st.sidebar.caption("EAR reference: open eye ≈ 0.25–0.35 · closed < 0.20")
st.sidebar.caption("MAR reference: resting ≈ 0.3–0.5 · yawning > 0.75")


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.title("👁️ DrowsAlert — Driver Fatigue Detection")
st.caption("Real-time drowsiness and yawning detection using EAR/MAR analysis · dlib 68-point landmarks")
st.divider()


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
try:
    detector, predictor = load_model()
    model_ok = True
except Exception as e:
    model_ok = False
    st.error(f"Could not load dlib model: {e}\n\n"
             "Ensure `shape_predictor_68_face_landmarks.dat` is in the same directory as `app.py`.")


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_webcam, tab_image, tab_about = st.tabs(["📷 Live Webcam", "🖼️ Image Analysis", "ℹ️ How It Works"])


# ── TAB 1: WEBCAM ────────────────────────────
with tab_webcam:
    st.caption("Click **Take Photo**, then see the analysis on the right.")

    col_cam, col_res = st.columns([3, 2], gap="large")

    with col_cam:
        camera_img = st.camera_input("Capture", label_visibility="collapsed")

    with col_res:
        if camera_img and model_ok:
            file_bytes = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            annotated, results = process_frame(
                frame.copy(), detector, predictor, ear_thresh, mar_thresh, draw_mesh
            )
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Analysed Frame", use_container_width=True)
            render_results(results, ear_thresh, mar_thresh)
        elif not model_ok:
            st.warning("Model not loaded.")
        else:
            st.info("Results will appear here after you take a photo.")



# ── TAB 2: IMAGE ANALYSIS ────────────────────
with tab_image:
    st.subheader("Upload Images for Analysis")

    uploaded = st.file_uploader(
        "Choose one or more images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
    )

    if uploaded and model_ok:
        st.caption(f"{len(uploaded)} file(s) loaded")

        for up_file in uploaded:
            st.subheader(up_file.name)

            file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is None:
                st.error(f"Could not decode {up_file.name}")
                continue

            annotated, results = process_frame(
                frame.copy(), detector, predictor, ear_thresh, mar_thresh, draw_mesh
            )
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            col_img, col_met = st.columns([3, 2], gap="large")
            with col_img:
                st.image(annotated_rgb, use_container_width=True, caption="Annotated")
                buf = io.BytesIO()
                Image.fromarray(annotated_rgb).save(buf, format="PNG")
                st.download_button(
                    "⬇️ Download annotated image",
                    data=buf.getvalue(),
                    file_name=f"annotated_{up_file.name}",
                    mime="image/png",
                )
            with col_met:
                render_results(results, ear_thresh, mar_thresh)

            st.divider()

    elif not model_ok:
        st.warning("Model not loaded.")
    else:
        st.info("Upload images above to begin analysis.")


# ── TAB 3: HOW IT WORKS ──────────────────────
with tab_about:
    st.subheader("How It Works")

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("**EAR — Eye Aspect Ratio**")
        st.markdown("""
Computed from 6 landmark points per eye:

```
EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)
```

- Open eye → EAR ≈ 0.25–0.35
- Closed eye → EAR drops near 0
- Sustained low EAR over multiple frames = drowsy alert

**MAR — Mouth Aspect Ratio**

Same approach applied to 20 mouth landmarks.
A high MAR indicates an open mouth = yawn.
Single-frame detection — no temporal window needed.
        """)

    with col_b:
        st.markdown("**Detection Pipeline**")
        st.markdown("""
1. Convert frame to grayscale
2. dlib HOG detector finds face bounding boxes
3. 68-point shape predictor localises facial geometry
4. Extract eye landmarks (36–47) and mouth landmarks (48–67)
5. Compute EAR and MAR per face
6. Apply thresholds → trigger alerts

**Landmark regions**

| Points | Region |
|--------|--------|
| 0–16   | Jawline |
| 36–41  | Left eye (EAR) |
| 42–47  | Right eye (EAR) |
| 48–67  | Mouth (MAR) |

**Stack:** OpenCV · dlib · NumPy · SciPy · Streamlit
        """)