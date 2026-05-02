import cv2
import dlib
import numpy as np
import os
import joblib
from classical_cv_pipeline import (
    localize_eye_rois,
    localize_mouth_roi,
    fused_eye_state,
    compute_mouth_aspect_ratio
)

# ===============================
# PARAMETERS
# ===============================
EYE_STATE_THRESH = 0.18
EYE_STATE_CONSEC_FRAMES = 20
MAR_THRESH = 0.6

detector = dlib.get_frontal_face_detector()

svm_model = None

def load_svm_model():
    """Load pre-trained eye state SVM model."""
    global svm_model
    if os.path.exists("eye_svm_model.pkl"):
        svm_model = joblib.load("eye_svm_model.pkl")
        print("✓ SVM model loaded")
    else:
        print("⚠ Warning: eye_svm_model.pkl not found. Train with train_eye_svm.py")

# running the code on an image (if available)

def run_on_dataset(folder_path):

    if svm_model is None:
        print("✗ SVM model not loaded. Cannot proceed.")
        return

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

    if not os.path.isdir(folder_path):
        print(f"✗ Folder not found: {folder_path}")
        return

    for file in os.listdir(folder_path):

        if not file.lower().endswith(valid_ext):
            continue

        img_path = os.path.join(folder_path, file)
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        print(f"\nProcessing Image: {file}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            print("→ No face detected")
            cv2.putText(frame, "no face detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        for i, face in enumerate(faces):
            fx, fy = face.left(), face.top()
            fw, fh = face.right() - face.left(), face.bottom() - face.top()
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (100, 255, 0), 2)

            eye_rois = localize_eye_rois(gray, fx, fy, fw, fh)
            mouth_roi = localize_mouth_roi(gray, fx, fy, fw, fh)

            eye_scores = []
            for (ex, ey, ew, eh) in eye_rois:
                eye_patch = frame[ey:ey + eh, ex:ex + ew]
                if eye_patch.size == 0:
                    continue
                score = fused_eye_state(eye_patch, svm_model)
                eye_scores.append(score)

            avg_eye_score = np.mean(eye_scores) if eye_scores else 0.5

            if mouth_roi is not None:
                mx, my, mw, mh = mouth_roi
                mouth_patch = frame[my:my + mh, mx:mx + mw]
                if mouth_patch.size > 0:
                    mar = compute_mouth_aspect_ratio(mouth_patch)
                else:
                    mar = 0.0
            else:
                mar = 0.0

            drowsy = avg_eye_score < EYE_STATE_THRESH
            yawning = mar > MAR_THRESH

            print(f"→ Face {i+1}:")
            print(f"   Eye openness: {avg_eye_score:.3f} (threshold: {EYE_STATE_THRESH})")
            print(f"   Drowsy: {'YES' if drowsy else 'NO'}")
            print(f"   MAR: {mar:.3f} (threshold: {MAR_THRESH})")
            print(f"   Yawning: {'YES' if yawning else 'NO'}")

            if drowsy:
                cv2.putText(frame, "eyes closed (drowsy)", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            if yawning:
                cv2.putText(frame, "yawning", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        cv2.imshow("Dataset Testing", frame)

        key = cv2.waitKey(500) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()

# running the code on webcam!
def run_webcam():

    if svm_model is None:
        print("✗ SVM model not loaded. Cannot proceed.")
        return

    COUNTER = 0
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            cv2.putText(frame, "no face detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        for face in faces:
            fx, fy = face.left(), face.top()
            fw, fh = face.right() - face.left(), face.bottom() - face.top()
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (100, 255, 0), 2)

            eye_rois = localize_eye_rois(gray, fx, fy, fw, fh)
            mouth_roi = localize_mouth_roi(gray, fx, fy, fw, fh)

            eye_scores = []
            for (ex, ey, ew, eh) in eye_rois:
                eye_patch = frame[ey:ey + eh, ex:ex + ew]
                if eye_patch.size == 0:
                    continue
                score = fused_eye_state(eye_patch, svm_model)
                eye_scores.append(score)

            avg_eye_score = np.mean(eye_scores) if eye_scores else 0.5

            if mouth_roi is not None:
                mx, my, mw, mh = mouth_roi
                mouth_patch = frame[my:my + mh, mx:mx + mw]
                if mouth_patch.size > 0:
                    mar = compute_mouth_aspect_ratio(mouth_patch)
                else:
                    mar = 0.0
            else:
                mar = 0.0

            if avg_eye_score < EYE_STATE_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_STATE_CONSEC_FRAMES:
                    cv2.putText(frame, "eyes closed (drowsy)", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                COUNTER = 0

            if mar > MAR_THRESH:
                cv2.putText(frame, "yawning", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        cv2.imshow("detection using webcam", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    load_svm_model()

    choice = str(input("A. webcam (enter 1)\nB. sample image (enter 2)\n: "))

    if choice == "1":
        run_webcam()

    elif choice == "2":
        folder = input("Enter dataset folder path: ")
        run_on_dataset(folder)

    else:
        print("Invalid choice")