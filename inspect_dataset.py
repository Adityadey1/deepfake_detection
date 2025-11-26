
import os
from pathlib import Path
from PIL import Image

root = Path(r"E:\code\deepfake_project\data\kaggle")
classes = []
for p in root.iterdir():
    if p.is_dir():
        classes.append(p.name)

info = {}
for cls in classes:
    folder = root / cls
    files = list(folder.glob("*"))
    # count common image extensions
    files = [f for f in files if f.suffix.lower() in ('.jpg','.jpeg','.png')]
    info[cls] = len(files)

print("Classes found:", classes)
for k,v in info.items():
    print(f"{k}: {v} images")

# print 10 sample paths per class
for cls in classes:
    folder = root / cls
    samples = [f for f in folder.glob("*") if f.suffix.lower() in ('.jpg','.jpeg','.png')]
    print(f"\nSamples from {cls}:")
    for s in samples[:10]:
        print(s.name)
