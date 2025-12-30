import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
from tensorflow.keras.applications.resnet import preprocess_input as res_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre

# ================= CONFIG =================
# Change ONLY this line to test different models:
MODEL_NAME = "efficientnet"   # options: efficientnet | resnet | mobilenet

MODEL_PATHS = {
    "efficientnet": r"E:\code\deepfake_project\results\efficientnet\model.keras",
    "resnet": r"E:\code\deepfake_project\results\resnet\model.keras",
    "mobilenet": r"E:\code\deepfake_project\results\mobilenet\model.keras",
}

DATASET_B = Path(r"E:\code\deepfake_project\data\dataset_B")
IMG_SIZE = 224
THRESHOLD = 0.5
# ==========================================


def preprocess(x):
    if MODEL_NAME == "efficientnet":
        return eff_pre(x)
    elif MODEL_NAME == "resnet":
        return res_pre(x)
    elif MODEL_NAME == "mobilenet":
        return mob_pre(x)
    else:
        raise ValueError("Invalid MODEL_NAME")


def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess(x)
    prob = model.predict(x, verbose=0)[0][0]
    return 1 if prob >= THRESHOLD else 0


def evaluate_folder(model, folder, true_label):
    correct = 0
    total = 0

    for img_path in folder.iterdir():
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        pred = predict_image(model, img_path)
        if pred == true_label:
            correct += 1
        total += 1

    return correct, total


def main():
    print(f"\nLoading model: {MODEL_NAME}")
    model = tf.keras.models.load_model(MODEL_PATHS[MODEL_NAME])

    real_correct, real_total = evaluate_folder(model, DATASET_B / "real", 0)
    fake_correct, fake_total = evaluate_folder(model, DATASET_B / "fake", 1)

    total_correct = real_correct + fake_correct
    total_images = real_total + fake_total
    accuracy = total_correct / total_images if total_images > 0 else 0

    print("\n=== Cross-Dataset Evaluation (Dataset-B) ===")
    print(f"Model          : {MODEL_NAME}")
    print(f"Real images    : {real_correct} / {real_total}")
    print(f"Fake images    : {fake_correct} / {fake_total}")
    print(f"Overall Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
