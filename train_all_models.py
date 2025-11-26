
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import random
import time
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

PROJECT_ROOT = Path(r"E:\code\deepfake_project")   
SPLIT_CSV = PROJECT_ROOT / "data" / "splits" / "dataset_split.csv"
CUSTOM_UNSEEN = PROJECT_ROOT / "data" / "custom_unseen"
RESULTS_DIR = PROJECT_ROOT / "results"
IMG_SIZE = 224
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["efficientnet", "mobilenet", "resnet", "all"], default="efficientnet")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--quick", action="store_true", help="Quick sanity run using smaller subsets")
    p.add_argument("--train_n", type=int, default=None, help="If quick, sample this many train examples")
    p.add_argument("--val_n", type=int, default=None, help="If quick, sample this many val examples")
    p.add_argument("--test_n", type=int, default=None, help="If quick, sample this many test examples")
    return p.parse_args()


def read_split_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df


def decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0..1
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img


def augment_image(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.08)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    # random crop jitter
    if tf.random.uniform([]) < 0.5:
        crop_size = tf.cast(tf.round(IMG_SIZE * 0.9), tf.int32)
        img = tf.image.random_crop(img, size=[crop_size, crop_size, 3])
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img


def preprocess_for_model(img, model_name):
    # img is float32 in 0..1
    x = img * 255.0
    if model_name == "efficientnet":
        return tf.keras.applications.efficientnet.preprocess_input(x)
    elif model_name == "mobilenet":
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)
    elif model_name == "resnet":
        return tf.keras.applications.resnet.preprocess_input(x)
    else:
        return x


def make_dataset(df_subset, model_name, batch_size, shuffle=False, augment=False):
    paths = df_subset["filepath"].tolist()
    labels = df_subset["label"].astype("float32").tolist()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    def _load(path, label):
        img = decode_and_resize(path)
        if augment:
            img = augment_image(img)
        img = preprocess_for_model(img, model_name)
        return img, label
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# ------------------ Model factory ------------------
def build_model(name="efficientnet", input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    if name == "efficientnet":
        base = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=input_shape, pooling="avg", weights="imagenet")
    elif name == "mobilenet":
        base = tf.keras.applications.MobileNetV2(include_top=False, input_shape=input_shape, pooling="avg", weights="imagenet")
    elif name == "resnet":
        base = tf.keras.applications.ResNet50(include_top=False, input_shape=input_shape, pooling="avg", weights="imagenet")
    else:
        raise ValueError("Unknown model")
    x = base.output
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=base.input, outputs=out)
    return model


# ------------------ Training helpers ------------------
def get_callbacks(experiment_name):
    out_dir = RESULTS_DIR / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = callbacks.ModelCheckpoint(str(out_dir / "best.h5"), monitor="val_loss", save_best_only=True)
    early = callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    csv_logger = callbacks.CSVLogger(str(out_dir / "training_log.csv"))
    return [ckpt, early, reduce_lr, csv_logger], out_dir


def compile_model(model, lr=1e-4):
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


def evaluate_and_save_results(model, ds_test, experiment_dir, prefix="test"):
    res = model.evaluate(ds_test, verbose=1)
    # res: [loss, acc, auc]
    keys = ["loss", "accuracy", "auc"]
    out = dict(zip(keys, res))
    # write to json
    with open(experiment_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as jf:
        json.dump(out, jf, indent=2)
    return out


def inference_on_unseen(model, model_name):
    unseen_paths = sorted([str(p) for p in Path(CUSTOM_UNSEEN).glob("*") if p.suffix.lower() in IMG_EXT])
    if not unseen_paths:
        print("No custom_unseen images found.")
        return None
    unseen_df = pd.DataFrame({"filepath": unseen_paths, "label": [1]*len(unseen_paths)})
    unseen_ds = make_dataset(unseen_df, model_name, batch_size=16, shuffle=False, augment=False)
    res = model.evaluate(unseen_ds, verbose=1)
    keys = ["loss", "accuracy", "auc"]
    return dict(zip(keys, res))


# ------------------ Run experiment ------------------
def run_experiment(model_name, args, run_all_dir=None):
    print("Loading split CSV:", SPLIT_CSV)
    df = read_split_csv(SPLIT_CSV)

    # optionally sample smaller sets for quick runs
    if args.quick:
        train_df = df[df.split == "train"].sample(n=min(args.train_n or 2000, df[df.split=="train"].shape[0]), random_state=SEED)
        val_df = df[df.split == "val"].sample(n=min(args.val_n or 500, df[df.split=="val"].shape[0]), random_state=SEED)
        test_df = df[df.split == "test"].sample(n=min(args.test_n or 1000, df[df.split=="test"].shape[0]), random_state=SEED)
    else:
        train_df = df[df.split == "train"].reset_index(drop=True)
        val_df = df[df.split == "val"].reset_index(drop=True)
        test_df = df[df.split == "test"].reset_index(drop=True)

    print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))
    train_ds = make_dataset(train_df, model_name, batch_size=args.batch_size, shuffle=True, augment=True)
    val_ds = make_dataset(val_df, model_name, batch_size=args.batch_size, shuffle=False, augment=False)
    test_ds = make_dataset(test_df, model_name, batch_size=args.batch_size, shuffle=False, augment=False)

    model = build_model(model_name)
    model = compile_model(model, lr=args.lr)
    print(model.summary())

    callbacks_list, exp_dir = get_callbacks(model_name if run_all_dir is None else run_all_dir)
    hist = model.fit(train_ds, epochs=args.epochs, validation_data=val_ds, callbacks=callbacks_list)
    # Save final model
    # Save final model in Keras format
    model.save(exp_dir / "model.keras", include_optimizer=False)
    # Also export as SavedModel directory (for TF Lite / serving if needed)
    model.export(exp_dir / "savedmodel")

    # Evaluate on test
    test_metrics = evaluate_and_save_results(model, test_ds, exp_dir, prefix="test")
    # Evaluate on custom_unseen
    unseen_metrics = inference_on_unseen(model, model_name)
    if unseen_metrics is not None:
        with open(exp_dir / "unseen_metrics.json", "w", encoding="utf-8") as jf:
            json.dump(unseen_metrics, jf, indent=2)
    # Save history
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(exp_dir / "history.csv", index=False)
    print("Experiment saved to:", exp_dir)
    return exp_dir


def main():
    args = parse_args()

    models_to_run = [args.model] if args.model != "all" else ["efficientnet", "mobilenet", "resnet"]

    for m in models_to_run:
        print("=== RUNNING:", m, "===")
        run_experiment(m, args)

    print("All done.")


if __name__ == "__main__":
    main()
