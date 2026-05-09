import streamlit as st
import cv2
import dlib
import numpy as np
from PIL import Image
import io
import joblib
from classical_cv_pipeline import (
    localize_eye_rois,
    localize_mouth_roi,
    fused_eye_state,
    compute_mouth_aspect_ratio
)

st.set_page_config(
    page_title="DrowsAlert · Driver Monitoring",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading face detector…")
def load_face_detector():
    return dlib.get_frontal_face_detector()

@st.cache_resource(show_spinner="Loading eye state classifier…")
def load_svm_model():
    try:
        svm = joblib.load("eye_svm_model.pkl")
        return svm
    except FileNotFoundError:
        st.error("⚠ eye_svm_model.pkl not found. Train with train_eye_svm.py")
        return None


# ─────────────────────────────────────────────
#  FRAME PROCESSING
# ─────────────────────────────────────────────
def process_frame(frame, detector, svm_model, eye_score_thresh, mar_thresh, draw_mesh=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    results = []

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame, results

    for i, face in enumerate(faces):
        fx, fy = face.left(), face.top()
        fw, fh = face.right() - face.left(), face.bottom() - face.top()
        eye_rois = localize_eye_rois(gray, fx, fy, fw, fh)
        mouth_roi = localize_mouth_roi(gray, fx, fy, fw, fh)

        eye_scores = []
        for (ex, ey, ew, eh) in eye_rois:
            eye_patch = frame[ey:ey + eh, ex:ex + ew]
            if eye_patch.size == 0:
                continue
            score = fused_eye_state(eye_patch, svm_model)
            eye_scores.append(score)
            if draw_mesh:
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

        avg_eye_score = np.mean(eye_scores) if eye_scores else 0.5

        if mouth_roi is not None:
            mx, my, mw, mh = mouth_roi
            mouth_patch = frame[my:my + mh, mx:mx + mw]
            if mouth_patch.size > 0:
                mar = compute_mouth_aspect_ratio(mouth_patch)
            else:
                mar = 0.0
            if draw_mesh:
                cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (255, 0, 0), 1)
        else:
            mar = 0.0

        drowsy = avg_eye_score < eye_score_thresh
        yawning = mar > mar_thresh

        results.append({"face": i + 1, "eye_score": avg_eye_score, "mar": mar,
                        "drowsy": drowsy, "yawning": yawning})

        if draw_mesh:
            box_color = (0, 0, 255) if drowsy else (0, 165, 255) if yawning else (0, 255, 0)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_color, 2)

        cv2.putText(frame, f"Eye Score: {avg_eye_score:.3f}", (10, 30),
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
def render_results(results, eye_score_thresh, mar_thresh):
    if not results:
        st.warning("No face detected in frame.")
        return

    for r in results:
        if len(results) > 1:
            st.subheader(f"Face {r['face']}")

        col1, col2 = st.columns(2)
        col1.metric("Eye Openness Score", f"{r['eye_score']:.3f}",
                    delta=f"threshold: {eye_score_thresh:.2f}", delta_color="off")
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

eye_score_thresh = st.sidebar.slider("Eye Openness Threshold (Drowsiness)", 0.05, 0.35, 0.18, 0.01,
                                      help="Eye score below this value → drowsy")
mar_thresh       = st.sidebar.slider("MAR Threshold (Yawning)",             0.40, 1.20, 0.60, 0.01,
                                      help="MAR above this value → yawning")
draw_mesh        = st.sidebar.checkbox("Draw ROI Boxes", value=True)

st.sidebar.divider()
st.sidebar.caption("Eye score: 0 (closed) to 1 (open) · threshold default: 0.18")
st.sidebar.caption("MAR reference: resting ≈ 0.3–0.5 · yawning > 0.60")


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.title("👁️ DrowsAlert — Driver Fatigue Detection")
st.caption("Real-time drowsiness and yawning detection using Haar cascades, HOG+SVM, and geometric analysis")
st.divider()


# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────
detector = load_face_detector()
svm_model = load_svm_model()
model_ok = svm_model is not None


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
                frame.copy(), detector, svm_model, eye_score_thresh, mar_thresh, draw_mesh
            )
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Analysed Frame", use_container_width=True)
            render_results(results, eye_score_thresh, mar_thresh)
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
                frame.copy(), detector, svm_model, eye_score_thresh, mar_thresh, draw_mesh
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
                render_results(results, eye_score_thresh, mar_thresh)

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
        st.markdown("**Eye Openness Detection**")
        st.markdown("""
Fused 3-method pipeline:

1. **Haar Cascade** (ROI localization)
   - Detects eyes in top 55% of face box

2. **HOG + LinearSVC** (primary classifier)
   - Extracts HOG features from 24×24 patches
   - Binary classifier: open/closed

3. **Canny + Ellipse** (geometric confidence)
   - Detects eye shape via edge detection
   - Blends with SVM score: 70% SVM + 30% geometry

**Final Score:** 0 (closed) to 1 (open)
- Open eye → 0.3–0.8
- Closed eye → 0.0–0.1
- Threshold: 0.18 → drowsy alert

**MAR — Mouth Aspect Ratio**

- Haar cascade detects mouth in bottom 40% of face
- Geometric contour analysis: height/width ratio
- Threshold: 0.6 → yawning alert
        """)

    with col_b:
        st.markdown("**Detection Pipeline**")
        st.markdown("""
1. Convert frame to grayscale
2. Cascade classifier finds face bounding boxes
3. Haar cascades localize eye and mouth ROIs
4. Fused pipeline computes eye openness score
5. Contour analysis computes MAR
6. Apply thresholds → trigger alerts

**Processing steps**

| Stage | Method |
|-------|--------|
| Face | dlib HOG+SVM detector |
| Eyes | Haar cascade |
| Mouth | Haar cascade |
| Eye state | HOG + SVM + ellipse |
| Mouth state | Contour geometry |

**Stack:** OpenCV · scikit-learn · scikit-image · NumPy · Streamlit
        """)