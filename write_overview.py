# scripts/write_overview.py
import csv
from pathlib import Path
from collections import defaultdict

root = Path(r"E:\code\deepfake_project\data\kaggle")
out_path = Path(r"E:\code\deepfake_project\data\splits\dataset_overview.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

# We'll look for directories named 'real' or 'fake' anywhere under root.
# If none found, fall back to immediate child dirs.
classes_to_count = defaultdict(int)
found_class_dirs = False

for d in root.rglob('*'):
    if d.is_dir():
        name = d.name.lower()
        if name in ('real', 'fake'):
            found_class_dirs = True
            # count images under this dir (recursively)
            count = sum(1 for f in d.rglob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png') and f.is_file())
            classes_to_count[name] += count

# Fallback: if no 'real'/'fake' directories found, count immediate subdirs as classes
if not found_class_dirs:
    for p in sorted([p for p in root.iterdir() if p.is_dir()]):
        count = sum(1 for f in p.rglob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png') and f.is_file())
        classes_to_count[p.name] = count

# Write CSV
rows = [["class", "count"]]
for k, v in sorted(classes_to_count.items()):
    rows.append([k, v])

with open(out_path, "w", newline="", encoding="utf-8") as wf:
    writer = csv.writer(wf)
    writer.writerows(rows)

print(f"Written {out_path} with {len(rows)-1} classes.")
for k, v in sorted(classes_to_count.items()):
    print(f"{k}: {v} images")
