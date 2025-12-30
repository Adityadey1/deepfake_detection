import argparse
from pathlib import Path
import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    EfficientNetB0,
    ResNet50,
    MobileNetV2
)

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path(r"E:\code\deepfake_project")
SPLIT_CSV = PROJECT_ROOT / "data" / "splits" / "dataset_split.csv"
IMG_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE
# ----------------------------------------


def build_model(model_name):
    if model_name == "efficientnet":
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    elif model_name == "resnet":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        preprocess = tf.keras.applications.resnet.preprocess_input
    elif model_name == "mobilenet":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    else:
        raise ValueError("Invalid model")

    base.trainable = False  # important for CPU training

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = preprocess(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


def load_dataset(df, batch_size):
    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((df["filepath"].values, df["label"].values))
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(1000).batch(batch_size).prefetch(AUTOTUNE)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["efficientnet", "resnet", "mobilenet"])
    parser.add_argument("--train_n", type=int, default=40000)
    parser.add_argument("--val_n", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    df = pd.read_csv(SPLIT_CSV)

    train_df = df[df.split == "train"].sample(n=args.train_n, random_state=42)
    val_df = df[df.split == "val"].sample(n=args.val_n, random_state=42)

    train_ds = load_dataset(train_df, args.batch_size)
    val_ds = load_dataset(val_df, args.batch_size)

    model = build_model(args.model)

    out_dir = PROJECT_ROOT / "results" / f"{args.model}_full"
    out_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
    )

    model.save(out_dir / "model.keras")
    print(f"Model saved to {out_dir}")


if __name__ == "__main__":
    main()
    