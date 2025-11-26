# scripts/create_dataset_split.py
import csv, json, random
from pathlib import Path

PROJECT_ROOT = Path(r"E:\code\deepfake_project")
KAGGLE_ROOT = PROJECT_ROOT / "data" / "kaggle"
CUSTOM_POOL = PROJECT_ROOT / "data" / "custom_fake"
CUSTOM_UNSEEN = PROJECT_ROOT / "data" / "custom_unseen"
OUT_CSV = PROJECT_ROOT / "data" / "splits" / "dataset_split.csv"
OUT_COUNTS = PROJECT_ROOT / "data" / "splits" / "dataset_counts.json"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)

def find_class_dirs(root: Path, class_name: str):
    return [p for p in root.rglob('*') if p.is_dir() and p.name.lower() == class_name.lower()]

def gather_images_from_dirs(dir_list):
    imgs = []
    for d in dir_list:
        imgs.extend([p.resolve() for p in d.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXT])
    return sorted(set(imgs))

def split_list(items, train_ratio, val_ratio):
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = items[:n_train]
    val = items[n_train:n_train+n_val]
    test = items[n_train+n_val:]
    return train, val, test

def main():
    real_dirs = find_class_dirs(KAGGLE_ROOT, "real")
    fake_dirs = find_class_dirs(KAGGLE_ROOT, "fake")

    real_imgs = gather_images_from_dirs(real_dirs)
    fake_imgs = gather_images_from_dirs(fake_dirs)

    custom_pool_imgs = sorted([p.resolve() for p in (CUSTOM_POOL).rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXT])
    custom_unseen_imgs = set([p.resolve() for p in (CUSTOM_UNSEEN).rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXT])
    custom_pool_imgs = [p for p in custom_pool_imgs if p not in custom_unseen_imgs]

    print(f"Found real dirs: {len(real_dirs)}  images: {len(real_imgs)}")
    print(f"Found fake dirs: {len(fake_dirs)}  images: {len(fake_imgs)}")
    print(f"Found custom_pool images: {len(custom_pool_imgs)}")
    print(f"Found custom_unseen reserved: {len(custom_unseen_imgs)}")

    fake_all = fake_imgs + custom_pool_imgs

    random.shuffle(real_imgs)
    random.shuffle(fake_all)

    real_train, real_val, real_test = split_list(real_imgs, TRAIN_RATIO, VAL_RATIO)
    fake_train, fake_val, fake_test = split_list(fake_all, TRAIN_RATIO, VAL_RATIO)

    rows = [["filepath","label","split"]]
    def add_rows(paths, label, split_name):
        for p in paths:
            rows.append([str(p), str(label), split_name])

    add_rows(real_train, 0, "train")
    add_rows(real_val, 0, "val")
    add_rows(real_test, 0, "test")
    add_rows(fake_train, 1, "train")
    add_rows(fake_val, 1, "val")
    add_rows(fake_test, 1, "test")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as wf:
        writer = csv.writer(wf)
        writer.writerows(rows)

    counts = {
        "real": {"train": len(real_train), "val": len(real_val), "test": len(real_test)},
        "fake": {"train": len(fake_train), "val": len(fake_val), "test": len(fake_test)},
        "custom_unseen_reserved": len(custom_unseen_imgs),
        "total": {"train": len(real_train)+len(fake_train), "val": len(real_val)+len(fake_val), "test": len(real_test)+len(fake_test)}
    }
    OUT_COUNTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_COUNTS, "w", encoding="utf-8") as jf:
        json.dump(counts, jf, indent=2)

    print("Wrote CSV:", OUT_CSV)
    print("Wrote counts summary:", OUT_COUNTS)
    print(json.dumps(counts, indent=2))

if __name__ == "__main__":
    main()
