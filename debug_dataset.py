# debug_dataset.py
import os, random
from PIL import Image, ImageDraw
from poly_utils import read_polys_txt

root = "datasets/synthtext_dbnet"
img_dir = os.path.join(root, "images")
ann_dir = os.path.join(root, "annotations")

names = [n for n in os.listdir(img_dir) if n.endswith(".jpg")]
for name in random.sample(names, 3):
    stem = os.path.splitext(name)[0]
    img = Image.open(os.path.join(img_dir, name)).convert("RGB")
    ann = os.path.join(ann_dir, stem + ".txt")
    polys = read_polys_txt(ann)
    draw = ImageDraw.Draw(img, "RGBA")
    for poly in polys:
        draw.polygon(poly, outline=(0,255,0,255), fill=(0,255,0,60))
    out = f"debug_{stem}.png"
    img.save(out)
    print("wrote", out, "polys:", len(polys))
