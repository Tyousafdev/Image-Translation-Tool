#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
from dbnet_text_detector import DBNet, DBDetector

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "outputs/dbnet_trained.pth"
INPUT_DIR = "test_pages"
OUTPUT_DIR = "detection_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "crops"), exist_ok=True)

# --- Load model ---
print("Loading model from", MODEL_PATH)
model = DBNet().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
detector = DBDetector(model, device=DEVICE)

# --- Run detection ---
all_results = {}

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith((".jpg",".jpeg",".png")): 
        continue

    path = os.path.join(INPUT_DIR, fname)
    print("Processing", path)
    img = Image.open(path).convert("RGB")

    results = detector.detect(img, max_side=1536, bin_thresh=0.4, unclip=1.5)
    all_results[fname] = results

    # Draw overlay
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    for r in results:
        poly = r["polygon"]
        if len(poly) >= 4:
            draw.line(poly + [poly[0]], width=3, fill=(255,0,0))
    overlay.save(os.path.join(OUTPUT_DIR, fname.replace(".jpg","_overlay.png")))

    # Crop each polygon's bounding box
    for i, r in enumerate(results):
        poly = r["polygon"]
        if len(poly) < 4: continue
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        x0,y0,x1,y1 = map(int, [min(xs),min(ys),max(xs),max(ys)])
        crop = img.crop((x0,y0,x1,y1))
        crop.save(os.path.join(OUTPUT_DIR, "crops", f"{fname}_box_{i:02d}.png"))

# --- Save JSON results ---
with open(os.path.join(OUTPUT_DIR, "detections.json"), "w") as f:
    json.dump(all_results, f, indent=2)

print("✅ Done — results saved in", OUTPUT_DIR)
