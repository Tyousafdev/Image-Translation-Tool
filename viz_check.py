#!/usr/bin/env python3
import os, random, cv2
import numpy as np

def draw_one(root, out_path="viz_debug.jpg"):
    img_dir = os.path.join(root, "images")
    ann_dir = os.path.join(root, "annotations")
    stems = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".png",".jpeg"))]
    if not stems:
        print("No images found."); return
    stem = random.choice(stems)
    img_path = None
    for ext in (".jpg",".jpeg",".png"):
        p = os.path.join(img_dir, stem+ext)
        if os.path.exists(p): img_path = p; break
    ann_path = os.path.join(ann_dir, stem+".txt")
    img = cv2.imread(img_path)
    vis = img.copy()
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            for ln in f:
                nums = list(map(int, ln.strip().split(",")))
                pts = np.array(nums, dtype=np.int32).reshape(-1,2)
                cv2.polylines(vis, [pts], True, (0,255,0), 2)
    cv2.imwrite(out_path, vis)
    print("Saved", out_path, "for", stem)

if __name__ == "__main__":
    # Try both datasets:
    draw_one("datasets/synthtext_dbnet", out_path="viz_synth.jpg")
    draw_one("datasets/manga109_dbnet", out_path="viz_manga.jpg")
