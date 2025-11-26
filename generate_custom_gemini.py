
import argparse
import csv
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from google import genai
from PIL import Image
from io import BytesIO

# ------------ config -------------
DEFAULT_MODEL = "gemini-2.5-flash-image"   # fast, good quality; change if you prefer another model
# ----------------------------------

def make_prompt(template, idx):
    # expand template if you want to include index, variation, etc.
    return template.format(idx=idx)

def save_image_from_part(part, out_path: Path):
    # part.as_image() returns a PIL.Image (per docs)
    image = part.as_image()
    image.save(out_path)
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100, help="Number of images to generate")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for generated images")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Gemini image model to use")
    parser.add_argument("--prompt_template", type=str,
                        default="Photorealistic portrait of a human face, close-up headshot, neutral expression, studio lighting, high detail, realistic skin texture, 3:4 aspect ratio. Variation {idx}",
                        help="Prompt template. Use {idx} to inject index.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir.parent / "custom_meta.csv"
    # if meta exists, append; otherwise create with header
    meta_exists = meta_path.exists()
    meta_file = open(meta_path, "a", newline="", encoding="utf-8")
    meta_writer = csv.writer(meta_file)
    if not meta_exists:
        meta_writer.writerow(["filename", "source", "prompt", "date", "notes"])

    # init client (reads GEMINI_API_KEY from env by default)
    import os
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


    for i in tqdm(range(1, args.count + 1), desc="Generating"):
        prompt = make_prompt(args.prompt_template, i)
        try:
            response = client.models.generate_content(
                model=args.model,
                contents=[prompt],
            )
        except Exception as e:
            print(f"Error generating image for idx {i}: {e}")
            continue

        # response.parts may contain text and/or inline_data (image)
        saved_any = False
        for part in response.parts:
            # text parts are allowed; image parts have inline_data / as_image()
            if getattr(part, "inline_data", None) is not None or getattr(part, "as_image", None) is not None:
                # use the documented helper method .as_image() (returns PIL Image)
                try:
                    filename = f"fake_gemini_{i:04d}.png"
                    out_path = out_dir / filename
                    # part.as_image() -> PIL.Image
                    pil_img = part.as_image()
                    pil_img.save(out_path)
                    meta_writer.writerow([filename, "gemini", prompt, datetime.utcnow().isoformat(), ""])
                    saved_any = True
                    break
                except Exception as e:
                    print(f"Failed to save image for idx {i}: {e}")
                    continue

        if not saved_any:
            # fallback: maybe part contains base64 inline_data; handle generically
            for part in response.parts:
                inline = getattr(part, "inline_data", None)
                if inline is not None:
                    data_b64 = inline.data
                    data = BytesIO(base64.b64decode(data_b64))
                    img = Image.open(data)
                    filename = f"fake_gemini_{i:04d}.png"
                    out_path = out_dir / filename
                    img.save(out_path)
                    meta_writer.writerow([filename, "gemini", prompt, datetime.utcnow().isoformat(), ""])
                    saved_any = True
                    break

        # small safety pause could be added here for rate limits
    meta_file.close()
    print("Done. Metadata appended to:", meta_path)

if __name__ == "__main__":
    main()
