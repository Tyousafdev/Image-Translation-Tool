#!/usr/bin/env python3
import os, scipy.io, numpy as np, cv2
from tqdm import tqdm

SRC_ROOT = os.path.expanduser("~/Downloads/SynthText/SynthText/SynthText")
OUT_ROOT = "datasets/synthtext_dbnet"
OUT_IMG = os.path.join(OUT_ROOT, "images")
OUT_ANN = os.path.join(OUT_ROOT, "annotations")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_ANN, exist_ok=True)

# How many samples to convert
LIMIT = 20000   # <- you can change this number

print("Loading gt.mat ...")
mat = scipy.io.loadmat(os.path.join(SRC_ROOT, "gt.mat"))
imnames = mat["imnames"][0]
wordBBs = mat["wordBB"][0]

count = min(LIMIT, len(imnames))
for i, (name, bb) in tqdm(enumerate(zip(imnames, wordBBs)), total=count):
    if i >= count:
        break
    relname = str(name[0])
    img_path = os.path.join(SRC_ROOT, relname)
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Save image
    out_img = os.path.join(OUT_IMG, f"{i:06d}.jpg")
    cv2.imwrite(out_img, img)

    # wordBB shape: (2,4,N) or (2,4)
    bb = np.array(bb)
    if bb.ndim == 2:
        bb = bb[:, :, None]
    bb = bb.transpose(2, 1, 0)  # (N,4,2)

    out_txt = os.path.join(OUT_ANN, f"{i:06d}.txt")
    with open(out_txt, "w") as f:
        for poly in bb:
            coords = [f"{int(x)},{int(y)}" for (x, y) in poly]
            f.write(",".join(coords) + "\n")

print(f"✅ Done. Converted {count} samples to datasets/synthtext_dbnet/")
