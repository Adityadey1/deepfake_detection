# scripts/count_kaggle_nested.py
from pathlib import Path

root = Path(r"E:\code\deepfake_project\data\kaggle")
print("Searching under:", root)

# find all directories named 'real' or 'fake' anywhere
reals = sorted([p for p in root.rglob('*') if p.is_dir() and p.name.lower() == 'real'])
fakes = sorted([p for p in root.rglob('*') if p.is_dir() and p.name.lower() == 'fake'])

def count_images(p):
    return sum(1 for _ in p.rglob('*') if _.is_file() and _.suffix.lower() in ('.jpg','.jpeg','.png'))

print("\nReal folders found:")
for p in reals:
    print("  ", p, "->", count_images(p), "images")

print("\nFake folders found:")
for p in fakes:
    print("  ", p, "->", count_images(p), "images")

total_real = sum(count_images(p) for p in reals)
total_fake = sum(count_images(p) for p in fakes)
print("\nTotals => real:", total_real, " fake:", total_fake)
