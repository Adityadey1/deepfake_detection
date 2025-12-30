from pathlib import Path

ROOT = Path(r"E:\code\deepfake_project\data\dataset_B")

real_imgs = list(ROOT.joinpath("real").glob("*.jpg"))
fake_imgs = list(ROOT.joinpath("fake").glob("*.jpg"))

print("Dataset-B summary")
print("------------------")
print(f"Real images: {len(real_imgs)}")
print(f"Fake images: {len(fake_imgs)}")

print("\nSample real images:")
for p in real_imgs[:5]:
    print(p.name)

print("\nSample fake images:")
for p in fake_imgs[:5]:
    print(p.name)
