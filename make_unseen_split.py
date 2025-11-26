import argparse
from pathlib import Path
import random
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--unseen_dir", required=True)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.source_dir)
    unseen = Path(args.unseen_dir)
    unseen.mkdir(parents=True, exist_ok=True)

    # collect all images
    files = sorted([p for p in src.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg')])

    random.seed(args.seed)
    k = int(len(files) * args.fraction)

    chosen = set(random.sample(files, k))

    # move files
    for p in chosen:
        p.rename(unseen / p.name)

    # update metadata
    meta_path = src.parent / "custom_meta.csv"
    if not meta_path.exists():
        print("No metadata at", meta_path)
        return

    # read old meta
    with open(meta_path, "r", newline="", encoding="utf-8") as rf:
        rows = list(csv.reader(rf))

    header = rows[0]
    if "split" not in header:
        header.append("split")

    new_rows = [header]

    for row in rows[1:]:
        filename = row[0]
        if (unseen / filename).exists():
            row.append("unseen_custom")
        else:
            row.append("custom_pool")
        new_rows.append(row)

    new_meta = src.parent / "custom_meta_with_split.csv"
    with open(new_meta, "w", newline="", encoding="utf-8") as wf:
        writer = csv.writer(wf)
        writer.writerows(new_rows)

    print(f"Moved {len(chosen)} files -> {unseen}")
    print(f"Updated metadata saved to: {new_meta}")

if __name__ == "__main__":
    main()

