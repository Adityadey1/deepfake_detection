#!/usr/bin/env python3
"""
scripts/inference.py

Run single-image inference using EfficientNet (default).

Examples:
---------
python scripts/inference.py --image path/to/image.jpg
python scripts/inference.py --image path/to/image.png --threshold 0.6
python scripts/inference.py --image face.jpg --show

Notes:
------
- Uses EfficientNet trained on Dataset-A
- Accepts any image resolution (.jpg / .png)
- Outputs REAL or FAKE with probability
"""

import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path(r"E:\code\deepfake_project")
MODEL_PATH = PROJECT_ROOT / "results" / "efficientnet" / "model.keras"
IMG_SIZE = 224
# ----------------------------------------


def preprocess_image(img_path: Path):
    """
    Load image, resize, and apply EfficientNet preprocessing
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    arr = np.asarray(img).astype("float32")
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)

    return np.expand_dims(arr, axis=0)  # add batch dimension


def load_model():
    """
    Load trained EfficientNet model
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    return tf.keras.models.load_model(MODEL_PATH)


def predict(model, input_tensor):
    """
    Run inference and return probability
    """
    preds = model.predict(input_tensor, verbose=0)

    # Handle different output shapes safely
    if preds.ndim == 2 and preds.shape[1] == 1:
        prob = float(preds[0, 0])
    elif preds.ndim == 1:
        prob = float(preds[0])
    else:
        prob = float(np.ravel(preds)[0])

    return prob


def pretty_print(prob, threshold):
    label = "FAKE (AI-generated)" if prob >= threshold else "REAL (human)"
    print("\n=== Inference Result ===")
    print(f"Model           : EfficientNet")
    print(f"Fake Probability: {prob:.4f}")
    print(f"Threshold       : {threshold}")
    print(f"Prediction      : {label}")


def main():
    parser = argparse.ArgumentParser(description="DeepFake Detection Inference (EfficientNet)")
    parser.add_argument("--image", required=True, help="Path to input image (.jpg / .png)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default=0.5)")
    parser.add_argument("--show", action="store_true", help="Display the input image")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"ERROR: Image not found: {img_path}")
        sys.exit(1)

    print("Loading model...")
    model = load_model()

    print("Processing image:", img_path)
    x = preprocess_image(img_path)

    prob = predict(model, x)
    pretty_print(prob, args.threshold)

    if args.show:
        try:
            Image.open(img_path).show()
        except Exception as e:
            print("Could not display image:", e)


if __name__ == "__main__":
    main()
