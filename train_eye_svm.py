"""
Train HOG + LinearSVC eye state classifier.

Downloads the MRL Eye Dataset automatically if eye_dataset/ is not present.
  MRL Eye Dataset: http://mrl.cs.vsb.cz/eyedataset
  84,898 images, PNG, grayscale infrared camera.

MRL filename format:
  s{subj}_{img}_{gender}_{glasses}_{condition}_{reflections}_{eyestate}_{sensor}.png
  eye_state field: 0 = closed, 1 = open

Usage:
  python train_eye_svm.py                  # downloads + trains
  python train_eye_svm.py --skip-download  # train from existing eye_dataset/
  python train_eye_svm.py --max 5000       # cap samples per class (default 5000)
"""

import argparse
import cv2
import numpy as np
import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.feature import hog
import joblib


# ===============================
# DATASET DOWNLOAD
# ===============================
MRL_URL  = "http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip"
MRL_ZIP  = "mrlEyes_2018_01.zip"
MRL_SIZE_HINT = "~4 GB"

def _progress_hook(count, block_size, total):
    downloaded = count * block_size
    if total > 0:
        pct = min(100, downloaded * 100 // total)
        mb = downloaded / 1_048_576
        total_mb = total / 1_048_576
        print(f"\r  {pct:3d}%  {mb:.0f} / {total_mb:.0f} MB", end="", flush=True)
    else:
        mb = downloaded / 1_048_576
        print(f"\r  {mb:.0f} MB downloaded", end="", flush=True)


def download_mrl(dest_zip=MRL_ZIP):
    print(f"Downloading MRL Eye Dataset ({MRL_SIZE_HINT}) …")
    print(f"  Source : {MRL_URL}")
    print(f"  Dest   : {dest_zip}")
    print("  This will take a while on a slow connection.")
    try:
        urllib.request.urlretrieve(MRL_URL, dest_zip, reporthook=_progress_hook)
        print()  # newline after progress
        return True
    except Exception as exc:
        print(f"\n✗ Download failed: {exc}")
        return False


def _parse_mrl_eye_state(stem):
    """
    Parse eye_state from MRL filename stem.
    Format A (8 fields): s0001_00001_gender_glasses_cond_refl_eyestate_sensor
    Format B (6 fields): s001_gender_glasses_eyestate_refl_sensor
    Returns 1 (open), 0 (closed), or None on parse failure.
    """
    parts = stem.split("_")
    try:
        if len(parts) >= 8:
            return int(parts[6])   # 8-field MRL 2018 format
        if len(parts) >= 6:
            return int(parts[3])   # 6-field shorter format
    except (ValueError, IndexError):
        pass
    return None


def extract_and_organise(zip_path, dataset_dir="eye_dataset", max_per_class=5000):
    """
    Extract MRL zip and sort images into dataset_dir/open and dataset_dir/closed.
    Skips files whose eye_state cannot be parsed.
    """
    open_dir   = Path(dataset_dir) / "open"
    closed_dir = Path(dataset_dir) / "closed"
    open_dir.mkdir(parents=True, exist_ok=True)
    closed_dir.mkdir(parents=True, exist_ok=True)

    n_open = n_closed = n_skip = 0
    print(f"\nExtracting and organising into {dataset_dir}/ …")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".png")]
        total = len(names)
        print(f"  {total} PNG images in archive")

        for idx, name in enumerate(names):
            if idx % 5000 == 0:
                print(f"  Processed {idx}/{total} …", flush=True)

            stem  = Path(name).stem
            state = _parse_mrl_eye_state(stem)

            if state == 1 and n_open < max_per_class:
                data = zf.read(name)
                dest = open_dir / Path(name).name
                dest.write_bytes(data)
                n_open += 1
            elif state == 0 and n_closed < max_per_class:
                data = zf.read(name)
                dest = closed_dir / Path(name).name
                dest.write_bytes(data)
                n_closed += 1
            else:
                n_skip += 1

            if n_open >= max_per_class and n_closed >= max_per_class:
                print("  Reached per-class cap, stopping early.")
                break

    print(f"  ✓ {n_open} open, {n_closed} closed extracted ({n_skip} skipped)")
    return n_open, n_closed


def ensure_dataset(dataset_dir="eye_dataset", max_per_class=5000):
    """Download and organise MRL dataset if eye_dataset/ is missing or empty."""
    open_dir   = Path(dataset_dir) / "open"
    closed_dir = Path(dataset_dir) / "closed"

    has_open   = open_dir.exists()   and any(open_dir.iterdir())
    has_closed = closed_dir.exists() and any(closed_dir.iterdir())

    if has_open and has_closed:
        print(f"✓ Dataset found at {dataset_dir}/")
        return True

    print(f"Dataset not found at {dataset_dir}/")

    zip_path = Path(MRL_ZIP)
    if not zip_path.exists():
        ok = download_mrl(str(zip_path))
        if not ok:
            return False
    else:
        print(f"✓ Found existing archive: {MRL_ZIP}")

    n_open, n_closed = extract_and_organise(str(zip_path), dataset_dir, max_per_class)
    if n_open == 0 or n_closed == 0:
        print("✗ Extraction produced no usable images — check archive integrity.")
        return False

    return True


# ===============================
# HOG FEATURE EXTRACTION
# ===============================
def extract_hog_features(patch, pixels_per_cell=(4, 4), cells_per_block=(2, 2)):
    """Extract HOG features from a 24x24 grayscale patch."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch

    # CLAHE normalisation for IR image contrast variation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    gray_resized = cv2.resize(gray, (24, 24))
    features = hog(
        gray_resized,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        feature_vector=True,
        channel_axis=None
    )
    return features


# ===============================
# DATASET LOADING
# ===============================
def load_dataset(dataset_path, max_per_class=None):
    """
    Load images from dataset_path/open and dataset_path/closed.
    Returns (X, y): X = HOG feature matrix, y = labels (1=open, 0=closed).
    """
    dataset_root = Path(dataset_path)
    open_dir     = dataset_root / "open"
    closed_dir   = dataset_root / "closed"

    if not open_dir.exists() or not closed_dir.exists():
        raise ValueError(
            f"Expected subdirectories 'open/' and 'closed/' in {dataset_root}"
        )

    X, y = [], []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}

    for label, folder in [(1, open_dir), (0, closed_dir)]:
        label_name = "open" if label == 1 else "closed"
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in valid_ext)
        if max_per_class:
            files = files[:max_per_class]

        print(f"Loading {label_name} eyes ({len(files)} files) …")
        loaded = 0
        for img_file in files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            X.append(extract_hog_features(img))
            y.append(label)
            loaded += 1

        print(f"  ✓ {loaded} loaded")

    if len(X) == 0:
        raise ValueError("No images loaded — check dataset directory.")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    n_open   = int((y == 1).sum())
    n_closed = int((y == 0).sum())
    print(f"\nTotal: {len(X)} samples | open: {n_open} | closed: {n_closed}")
    print(f"Feature dimension: {X.shape[1]}")
    return X, y


# ===============================
# TRAIN + EVALUATE
# ===============================
def train_model(X, y, C=1.0, max_iter=2000):
    """Train LinearSVC with 80/20 train-test split, print classification report."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTraining LinearSVC on {len(X_tr)} samples …")
    model = LinearSVC(C=C, max_iter=max_iter, dual=False, random_state=42)
    model.fit(X_tr, y_tr)

    train_acc = model.score(X_tr, y_tr)
    test_acc  = model.score(X_te, y_te)
    print(f"Train accuracy : {train_acc:.3f}")
    print(f"Test  accuracy : {test_acc:.3f}")
    print("\nClassification report (test set):")
    print(classification_report(y_te, model.predict(X_te),
                                target_names=["closed", "open"]))

    if test_acc < 0.75:
        print("⚠ Test accuracy below 75% — consider a larger or higher quality dataset.")

    return model, test_acc


def save_model(model, output_path="eye_svm_model.pkl"):
    joblib.dump(model, output_path)
    print(f"✓ Model saved to: {output_path}")


# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HOG+SVM eye state classifier")
    parser.add_argument("dataset", nargs="?", default="eye_dataset",
                        help="Path to dataset directory (default: eye_dataset)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip auto-download, use existing dataset directory")
    parser.add_argument("--max", type=int, default=5000, metavar="N",
                        help="Max images per class to use (default: 5000)")
    args = parser.parse_args()

    print("=" * 55)
    print("EYE STATE SVM TRAINER")
    print("=" * 55)

    if not args.skip_download:
        ok = ensure_dataset(args.dataset, max_per_class=args.max)
        if not ok:
            print("\n✗ Dataset setup failed. Aborting.")
            sys.exit(1)

    try:
        X, y = load_dataset(args.dataset, max_per_class=args.max)
        model, acc = train_model(X, y)
        save_model(model)
        print(f"\n{'✓' if acc >= 0.75 else '⚠'} Done (test accuracy: {acc:.3f})")
    except Exception as exc:
        print(f"\n✗ Error: {exc}")
        sys.exit(1)
