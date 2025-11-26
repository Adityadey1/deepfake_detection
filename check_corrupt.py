# scripts/check_corrupt.py
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import csv

root = Path(r"E:\code\deepfake_project\data\kaggle\real_vs_fake\real-vs-fake\test.csv")
corrupt_list = []

for cls in [p for p in root.iterdir() if p.is_dir()]:
    for img_path in cls.glob("*"):
        if img_path.suffix.lower() not in ('.jpg','.jpeg','.png'):
            continue
        try:
            with Image.open(img_path) as im:
                im.verify()
        except (UnidentifiedImageError, OSError, Exception) as e:
            corrupt_list.append((str(img_path), cls.name, str(e)))

print("Corrupt images found:", len(corrupt_list))
if corrupt_list:
    with open(root / "corrupt_images.csv", "w", newline='', encoding='utf-8') as wf:
        import csv
        writer = csv.writer(wf)
        writer.writerow(["path","class","error"])
        writer.writerows(corrupt_list)
    print("Saved corrupt_images.csv")
