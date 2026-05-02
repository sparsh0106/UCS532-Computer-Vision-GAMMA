import cv2
import numpy as np
from skimage.feature import hog

# ===============================
# HAAR CASCADE INITIALIZATION
# ===============================
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
)


# ===============================
# STEP 1: HAAR CASCADE ROI LOCALIZATION
# ===============================
def localize_eye_rois(gray_face, face_x, face_y, face_w, face_h):
    """
    Detect eyes in top 55% of face box.
    Returns list of (x, y, w, h) in image coordinates.
    """
    top_55_h = int(face_h * 0.55)
    face_roi = gray_face[face_y:face_y + top_55_h, face_x:face_x + face_w]

    eyes = eye_cascade.detectMultiScale(
        face_roi,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10, 10)
    )

    eye_rois = []
    for (ex, ey, ew, eh) in eyes:
        eye_rois.append((face_x + ex, face_y + ey, ew, eh))

    return eye_rois


def localize_mouth_roi(gray_face, face_x, face_y, face_w, face_h):
    """
    Detect mouth in bottom 40% of face box.
    Returns (x, y, w, h) in image coordinates or None.
    """
    bottom_40_y = int(face_h * 0.60)
    mouth_roi_h = int(face_h * 0.40)

    face_roi = gray_face[
        face_y + bottom_40_y:face_y + face_h,
        face_x:face_x + face_w
    ]

    smiles = smile_cascade.detectMultiScale(
        face_roi,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(20, 20)
    )

    if len(smiles) > 0:
        sx, sy, sw, sh = smiles[0]
        return (face_x + sx, face_y + bottom_40_y + sy, sw, sh)

    return None


# ===============================
# STEP 2: HOG + LinearSVC CLASSIFIER
# ===============================
def extract_hog_features(patch, pixels_per_cell=(4, 4), cells_per_block=(2, 2)):
    """
    Extract HOG features from a grayscale patch.
    Expected patch: 24x24 grayscale.
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
    gray_resized = cv2.resize(gray, (24, 24))

    features = hog(
        gray_resized,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        feature_vector=True,
        channel_axis=None
    )

    return features


def classify_eye_state(patch, svm_model):
    """
    Classify eye as open/closed using HOG + SVM.
    Returns (is_open, confidence_score).
    is_open: 1 = open, 0 = closed
    confidence_score: decision function output (higher = more confident it's open)
    """
    features = extract_hog_features(patch)
    features = features.reshape(1, -1)

    prediction = svm_model.predict(features)[0]
    score = svm_model.decision_function(features)[0]

    return int(prediction), score


# ===============================
# STEP 3: CANNY + ELLIPSE GEOMETRIC CONFIDENCE
# ===============================
def compute_ellipse_confidence(patch):
    """
    Detect eye shape via Canny + fitEllipse.
    Returns minor/major axis ratio (0-1), or None if ellipse fitting fails.
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    if len(largest_contour) < 5:
        return None

    try:
        ellipse = cv2.fitEllipse(largest_contour)
        (cx, cy), (ma, mi), angle = ellipse

        if ma > 0:
            ratio = mi / ma
            return ratio
    except:
        return None

    return None


# ===============================
# FUSED EYE STATE SCORING
# ===============================
def fused_eye_state(patch, svm_model, svm_weight=0.7, ellipse_weight=0.3):
    """
    Combine SVM confidence + ellipse geometry.
    Returns final_score (0-1, higher = more open).
    """
    svm_pred, svm_score = classify_eye_state(patch, svm_model)

    svm_normalized = (svm_score + 1.0) / 2.0
    svm_normalized = np.clip(svm_normalized, 0, 1)

    ellipse_ratio = compute_ellipse_confidence(patch)

    if ellipse_ratio is None:
        return svm_normalized

    final_score = (svm_weight * svm_normalized) + (ellipse_weight * ellipse_ratio)
    return np.clip(final_score, 0, 1)


# ===============================
# MAR COMPUTATION (GEOMETRIC ONLY)
# ===============================
def compute_mouth_aspect_ratio(mouth_patch):
    """
    Compute mouth aspect ratio from image patch.
    Returns MAR value (0-2.0, higher = more open).
    mouth_patch: BGR or grayscale image array
    """
    if mouth_patch is None or mouth_patch.size == 0:
        return 0.0

    gray_mouth = cv2.cvtColor(mouth_patch, cv2.COLOR_BGR2GRAY) if len(mouth_patch.shape) == 3 else mouth_patch

    blurred = cv2.GaussianBlur(gray_mouth, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0.0

    largest_contour = max(contours, key=cv2.contourArea)

    if len(largest_contour) < 4:
        return 0.0

    rect = cv2.boundingRect(largest_contour)
    rect_w, rect_h = rect[2], rect[3]

    if rect_w > 0:
        mar = rect_h / (rect_w + 1e-6)
        return np.clip(mar, 0, 2.0)

    return 0.0
