
import argparse
import os
from pathlib import Path
import time
import csv
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import pandas as pd

# -------------- Config / Args --------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="efficientnetb0",
                    choices=["efficientnetb0", "mobilenetv2", "resnet50"],
                    help="Backbone model")
parser.add_argument("--csv", type=str,
                    default=r"E:\code\deepfake_project\data\splits\dataset_split.csv",
                    help="Path to dataset_split.csv")
parser.add_argument("--project_root", type=str,
                    default=r"E:\code\deepfake_project",
                    help="Project root")
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision")
parser.add_argument("--save_dir", type=str,
                    default=r"E:\code\deepfake_project\results",
                    help="Where to save models and logs")
args = parser.parse_args()

PROJECT_ROOT = Path(args.project_root)
CSV_PATH = Path(args.csv)
CUSTOM_UNSEEN = PROJECT_ROOT / "data" / "custom_unseen"
SAVE_DIR = Path(args.save_dir)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = args.img_size
BATCH_SIZE = args.batch
AUTOTUNE = tf.data.AUTOTUNE

if args.mixed_precision:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    print("[INFO] Mixed precision enabled.")

# -------------- Helpers --------------
def read_split_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Expect columns: filepath,label,split
    return df

def path_label_from_row(row):
    return row["filepath"], int(row["label"])

def decode_and_resize(path, label, img_size=IMG_SIZE):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0-1
    img = tf.image.resize(img, [img_size, img_size])
    return img, label

# Train augmentation function
def augment(img, label):
    # Random flip
    img = tf.image.random_flip_left_right(img)
    # Random brightness/contrast
    img = tf.image.random_brightness(img, 0.08)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    # Random hue / saturation small changes
    img = tf.image.random_saturation(img, 0.95, 1.05)
    # Random crop -> resize back (simulates RandomResizedCrop)
    crop_scale = tf.random.uniform([], 0.85, 1.0)
    h = tf.cast(tf.shape(img)[0] * crop_scale, tf.int32)
    w = tf.cast(tf.shape(img)[1] * crop_scale, tf.int32)
    img = tf.image.random_crop(img, size=[h, w, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    # gaussian noise
    if tf.random.uniform([]) < 0.15:
        noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.01)
        img = img + noise
        img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

def prepare_dataset(df_split, split_name, batch_size=BATCH_SIZE, shuffle=False, augment_flag=False):
    df_sub = df_split[df_split["split"] == split_name]
    filepaths = df_sub["filepath"].tolist()
    labels = df_sub["label"].astype(int).tolist()
    if len(filepaths) == 0:
        return None
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(lambda p, l: (tf.cast(p, tf.string), tf.cast(l, tf.int32)),
                num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda p, l: tf.py_function(func=lambda p_, l_: decode_and_resize(p_.numpy().decode("utf-8"), int(l_), IMG_SIZE),
                                            inp=[p, l],
                                            Tout=(tf.float32, tf.int32)),
                num_parallel_calls=AUTOTUNE)
    # The py_function returns tensors with unknown shapes; set static shapes
    def set_shape(img, label):
        img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        label.set_shape([])
        return img, label

    ds = ds.map(set_shape, num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=2048, reshuffle_each_iteration=True)
    if augment_flag:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds

# -------------- Model factory --------------
def get_backbone(name, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    name = name.lower()
    if name == "efficientnetb0" or name == "efficientnetb0" or name == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
    elif name == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
    elif name == "resnet50":
        base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
    else:
        raise ValueError("Unknown model " + name)
    return base

def build_model(name):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # preprocessing (use model-specific preprocessing)
    if name == "efficientnetb0":
        x = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)
    elif name == "mobilenetv2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    elif name == "resnet50":
        x = tf.keras.applications.resnet.preprocess_input(inputs * 255.0)
    else:
        x = inputs

    base = get_backbone(name, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False  # start with frozen backbone
    x = base(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    if args.mixed_precision:
        # when using mixed precision, final layer dtype must be float32 for stability
        x = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    else:
        x = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, x, name=f"{name}_binary")
    return model

# -------------- Main --------------
def compute_class_weights(df):
    # compute weights on training set
    train_df = df[df["split"] == "train"]
    counts = train_df["label"].value_counts().to_dict()
    # label 0 and 1
    n0 = counts.get(0, 0)
    n1 = counts.get(1, 0)
    total = n0 + n1
    if n0 == 0 or n1 == 0:
        return None
    w0 = total / (2 * n0)
    w1 = total / (2 * n1)
    return {0: w0, 1: w1}

def evaluate_and_report(model, df, split_name, out_prefix):
    sub = df[df["split"] == split_name]
    if len(sub) == 0:
        print(f"[WARN] No samples for split {split_name}")
        return {}
    filepaths = sub["filepath"].tolist()
    labels = sub["label"].astype(int).tolist()

    preds = []
    batch = []
    B = 32
    t0 = time.time()
    for i in range(0, len(filepaths), B):
        batch_paths = filepaths[i:i+B]
        imgs = []
        for p in batch_paths:
            img = tf.io.read_file(p)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            imgs.append(img.numpy())
        imgs = np.stack(imgs, axis=0)
        probs = model.predict(imgs, verbose=0)
        preds.extend([float(p[0]) for p in probs])
    t1 = time.time()
    elapsed = t1 - t0
    inf_per_image = elapsed / max(1, len(filepaths))
    # binarize at 0.5
    y_pred = [1 if p >= 0.5 else 0 for p in preds]
    acc = accuracy_score(labels, y_pred)
    try:
        auc = roc_auc_score(labels, preds)
    except Exception:
        auc = float("nan")
    prec, recall, f1, _ = precision_recall_fscore_support(labels, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(labels, y_pred)
    # write prediction CSV
    out_csv = SAVE_DIR / f"{out_prefix}_{split_name}_preds.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as wf:
        writer = csv.writer(wf)
        writer.writerow(["filepath", "label", "pred_prob", "pred_label"])
        for p, y, prob in zip(filepaths, labels, preds):
            writer.writerow([p, y, prob, 1 if prob >= 0.5 else 0])

    results = {
        "acc": float(acc),
        "auc": float(auc) if not np.isnan(auc) else None,
        "precision": float(prec),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "inference_time_per_image_s": float(inf_per_image),
        "preds_csv": str(out_csv)
    }
    print(f"Eval {out_prefix} {split_name}: acc={acc:.4f} auc={auc:.4f} prec={prec:.4f} rec={recall:.4f} f1={f1:.4f} inf_s/img={inf_per_image:.4f}")
    return results

def eval_on_custom_unseen(model):
    # make a simple df from files in custom_unseen
    files = sorted([str(p) for p in (CUSTOM_UNSEEN).glob("*") if p.suffix.lower() in (".jpg",".jpeg",".png")])
    if len(files) == 0:
        print("[WARN] No custom_unseen images found.")
        return {}
    labels = [1]*len(files)  # all fake
    preds = []
    B = 32
    t0 = time.time()
    for i in range(0, len(files), B):
        batch_paths = files[i:i+B]
        imgs = []
        for p in batch_paths:
            img = tf.io.read_file(p)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            imgs.append(img.numpy())
        imgs = np.stack(imgs, axis=0)
        probs = model.predict(imgs, verbose=0)
        preds.extend([float(p[0]) for p in probs])
    t1 = time.time()
    elapsed = t1 - t0
    inf_per_image = elapsed / max(1, len(files))
    y_pred = [1 if p >= 0.5 else 0 for p in preds]
    acc = accuracy_score(labels, y_pred)
    try:
        auc = roc_auc_score(labels, preds)
    except Exception:
        auc = float("nan")
    prec, recall, f1, _ = precision_recall_fscore_support(labels, y_pred, average="binary", zero_division=0)
    print(f"Custom unseen eval: acc={acc:.4f} auc={auc:.4f} prec={prec:.4f} rec={recall:.4f} f1={f1:.4f} inf_s/img={inf_per_image:.4f}")
    out_csv = SAVE_DIR / f"{args.model}_custom_unseen_preds.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as wf:
        writer = csv.writer(wf)
        writer.writerow(["filepath", "label", "pred_prob", "pred_label"])
        for p, prob in zip(files, preds):
            writer.writerow([p, 1, prob, 1 if prob >= 0.5 else 0])
    return {"acc": acc, "auc": auc, "precision": prec, "recall": recall, "f1": f1, "preds_csv": str(out_csv)}

def main():
    print("Loading CSV:", CSV_PATH)
    df = read_split_csv(CSV_PATH)
    # ensure splits and labels are correct dtype
    df["label"] = df["label"].astype(int)
    df["split"] = df["split"].astype(str)

    train_ds = prepare_dataset(df, "train", batch_size=BATCH_SIZE, shuffle=True, augment_flag=True)
    val_ds = prepare_dataset(df, "val", batch_size=BATCH_SIZE, shuffle=False, augment_flag=False)
    test_ds = prepare_dataset(df, "test", batch_size=BATCH_SIZE, shuffle=False, augment_flag=False)

    print("Train/Val/Test sizes (approx batches):", 
          None if train_ds is None else "train_batches="+str(len(df[df["split"]=="train"])//BATCH_SIZE),
          None if val_ds is None else "val_batches="+str(len(df[df["split"]=="val"])//BATCH_SIZE),
          None if test_ds is None else "test_batches="+str(len(df[df["split"]=="test"])//BATCH_SIZE))

    # class weights
    class_weights = compute_class_weights(df)
    print("Class weights:", class_weights)

    # build model
    model = build_model(args.model)
    model.summary()

    opt = optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    # callbacks
    run_id = f"{args.model}_{IMG_SIZE}px_{int(time.time())}"
    ckpt_path = SAVE_DIR / f"{run_id}_best.h5"
    cb_ckpt = callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_accuracy", save_best_only=True, mode="max")
    cb_es = callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)
    cb_rlr = callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-7)
    cb_csv = callbacks.CSVLogger(SAVE_DIR / f"{run_id}_training_log.csv")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[cb_ckpt, cb_es, cb_rlr, cb_csv],
        class_weight=class_weights,
    )

    # save final model
    final_saved = SAVE_DIR / f"{run_id}_savedmodel"
    model.save(final_saved)
    print("Saved final model to:", final_saved)

    # Evaluate on test
    res_test = evaluate_and_report(model, df, "test", out_prefix=run_id)
    # Evaluate on custom unseen
    res_unseen = eval_on_custom_unseen(model)

    # Save summary
    summary = {
        "run_id": run_id,
        "model": args.model,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": args.epochs,
        "lr": args.lr,
        "test_eval": res_test,
        "custom_unseen_eval": res_unseen
    }
    with open(SAVE_DIR / f"{run_id}_summary.json", "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)

    print("Training run complete. Summary saved.")

if __name__ == "__main__":
    main()
