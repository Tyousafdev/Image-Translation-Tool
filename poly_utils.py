# poly_utils.py
import json, os
import numpy as np
from PIL import Image, ImageDraw

def read_polys_txt(txt_path):
    """Each line: x1,y1,x2,y2,...,xn,yn  (any n>=4, no header)."""
    polys = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            nums = [float(x) for x in line.replace(",", " ").split()]
            if len(nums) < 8 or len(nums) % 2 != 0:
                continue
            xy = [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]
            polys.append(xy)
    return polys

def read_polys_labelme(json_path):
    """LabelMe .json with 'shapes'-> list of {'points': [[x,y],...], 'label':...}"""
    with open(json_path, "r") as f:
        data = json.load(f)
    polys = []
    for s in data.get("shapes", []):
        pts = s.get("points", [])
        if len(pts) >= 4:
            polys.append([(float(x), float(y)) for x,y in pts])
    return polys

def rasterize_polys(polys, H, W, blur_px=0):
    """Return uint8 mask in {0,1} of size (H,W). Optional blur in caller for soft targets."""
    if len(polys) == 0:
        return np.zeros((H, W), dtype=np.uint8)
    mask_img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask_img)
    for poly in polys:
        if len(poly) >= 3:
            d.polygon(poly, outline=1, fill=1)
    return np.array(mask_img, dtype=np.uint8)
