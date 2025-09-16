#!/usr/bin/env python3
# DBNet-like text detector prototype (Step 1)
# See README in header comments of the notebook cell that generated this.
import os, json, math, random
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

seed_all(1234)

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.bn(self.conv(x)))

class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(ConvBNReLU(3, 32, 3, 2, 1), ConvBNReLU(32, 32, 3, 1, 1))
        self.layer2 = nn.Sequential(ConvBNReLU(32, 64, 3, 2, 1), ConvBNReLU(64, 64, 3, 1, 1))
        self.layer3 = nn.Sequential(ConvBNReLU(64, 128, 3, 2, 1), ConvBNReLU(128, 128, 3, 1, 1))
        self.layer4 = nn.Sequential(ConvBNReLU(128, 256, 3, 2, 1), ConvBNReLU(256, 256, 3, 1, 1))
    def forward(self, x):
        c1 = self.layer1(x); c2 = self.layer2(c1); c3 = self.layer3(c2); c4 = self.layer4(c3)
        return c1, c2, c3, c4

class FPNNeck(nn.Module):
    def __init__(self, in_channels=(32, 64, 128, 256), out_c=128):
        super().__init__()
        c1, c2, c3, c4 = in_channels
        self.lat4 = nn.Conv2d(c4, out_c, 1)
        self.lat3 = nn.Conv2d(c3, out_c, 1)
        self.lat2 = nn.Conv2d(c2, out_c, 1)
        self.lat1 = nn.Conv2d(c1, out_c, 1)
        self.smooth3 = ConvBNReLU(out_c, out_c, 3, 1, 1)
        self.smooth2 = ConvBNReLU(out_c, out_c, 3, 1, 1)
        self.smooth1 = ConvBNReLU(out_c, out_c, 3, 1, 1)
    def forward(self, c1, c2, c3, c4):
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.lat2(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')
        p1 = self.lat1(c1) + F.interpolate(p2, scale_factor=2, mode='nearest')
        p3 = self.smooth3(p3); p2 = self.smooth2(p2); p1 = self.smooth1(p1)
        return p1, p2, p3, p4

class DBHead(nn.Module):
    def __init__(self, in_c=128, k=50):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            ConvBNReLU(in_c*4, in_c, 3, 1, 1),
            ConvBNReLU(in_c, in_c//2, 3, 1, 1)
        )
        self.p_out = nn.Conv2d(in_c//2, 1, 1)
        self.t_out = nn.Conv2d(in_c//2, 1, 1)
    def forward(self, p1, p2, p3, p4):
        p2u = F.interpolate(p2, size=p1.shape[-2:], mode='bilinear', align_corners=False)
        p3u = F.interpolate(p3, size=p1.shape[-2:], mode='bilinear', align_corners=False)
        p4u = F.interpolate(p4, size=p1.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([p1, p2u, p3u, p4u], dim=1)
        x = self.conv(x)
        # 🔥 return raw logits, not passed through sigmoid
        P = self.p_out(x)
        T = self.t_out(x)
        B = self.k * (P - T)
        return P, T, B

class DBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SimpleBackbone()
        self.neck = FPNNeck()
        self.head = DBHead()
    def forward(self, x):
        c1,c2,c3,c4 = self.backbone(x)
        p1,p2,p3,p4 = self.neck(c1,c2,c3,c4)
        return self.head(p1,p2,p3,p4)

def make_blurred_mask_target(mask: torch.Tensor, blur=3) -> torch.Tensor:
    target_list = []
    m = mask.detach().cpu().numpy()
    for i in range(m.shape[0]):
        arr = (m[i,0]*255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L").filter(ImageFilter.GaussianBlur(radius=blur))
        arr2 = np.array(img).astype(np.float32)/255.0
        target_list.append(arr2[None, ...])
    t = torch.from_numpy(np.stack(target_list, axis=0))
    return t.to(mask.device)

class DBLoss(nn.Module):
    def __init__(self, bce_weight=1.0, l1_weight=1.0):
        super().__init__()
        # ✅ use BCEWithLogitsLoss (safe for autocast)
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.bce_weight = bce_weight
        self.l1_weight = l1_weight
    def forward(self, P, T, B, gt_mask):
        T_target = make_blurred_mask_target(gt_mask, blur=3)
        loss_p = self.bce(P, gt_mask)
        loss_t = self.l1(torch.sigmoid(T), T_target)  # still compare after sigmoid
        loss_b = self.bce(B, gt_mask)
        loss = self.bce_weight*loss_p + self.l1_weight*loss_t + 0.5*loss_b
        return loss, {
            "loss_p": loss_p.item(),
            "loss_t": loss_t.item(),
            "loss_b": loss_b.item()
        }


def connected_components(binary: np.ndarray) -> List[np.ndarray]:
    H, W = binary.shape
    visited = np.zeros_like(binary, dtype=bool); comps = []
    for y in range(H):
        for x in range(W):
            if binary[y, x] and not visited[y, x]:
                q = [(y,x)]; visited[y,x]=True; coords=[]
                while q:
                    cy,cx = q.pop()
                    coords.append((cy,cx))
                    for ny,nx in [(cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)]:
                        if 0<=ny<H and 0<=nx<W and binary[ny,nx] and not visited[ny,nx]:
                            visited[ny,nx]=True; q.append((ny,nx))
                mask = np.zeros_like(binary, dtype=bool)
                ys,xs = zip(*coords); mask[np.array(ys), np.array(xs)] = True
                comps.append(mask)
    return comps

def mask_to_quad(mask: np.ndarray) -> List[Tuple[float,float]]:
    ys, xs = np.where(mask)
    if len(xs)==0: return []
    x0,x1 = xs.min(), xs.max(); y0,y1 = ys.min(), ys.max()
    return [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]

def unclip_quad(quad, ratio, H, W):
    if not quad: return quad
    cx = sum([p[0] for p in quad])/4.0; cy = sum([p[1] for p in quad])/4.0
    new = []
    for (x,y) in quad:
        nx = cx + (x-cx)*ratio; ny = cy + (y-cy)*ratio
        nx = max(0, min(W-1, nx)); ny = max(0, min(H-1, ny)); new.append((nx,ny))
    return new

def score_instance(prob_map: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum()==0: return 0.0
    return float(prob_map[mask].mean())

class DBDetector:
    def __init__(self, model: nn.Module, device: torch.device = device):
        self.model = model.to(device); self.model.eval(); self.device = device
    @torch.no_grad()
    def detect(self, image_pil: Image.Image, max_side: int = 768, bin_thresh: float = 0.4, unclip: float = 1.5):
        img = image_pil.convert("RGB"); W,H = img.size
        scale = min(max_side / max(W,H), 1.0)
        if scale < 1.0:
            newW,newH = int(W*scale), int(H*scale); img = img.resize((newW,newH), Image.BILINEAR)
        else:
            newW,newH = W,H
        x = torch.from_numpy(np.array(img).transpose(2,0,1)).float()[None,...]/255.0
        x = x.to(self.device)
        P,T,B = self.model(x)
        p_map = F.interpolate(P, size=(newH,newW), mode='bilinear', align_corners=False)[0,0].detach().cpu().numpy()
        b_map = F.interpolate(B, size=(newH,newW), mode='bilinear', align_corners=False)[0,0].detach().cpu().numpy()
        binary = (b_map > bin_thresh).astype(np.uint8)
        comps = connected_components(binary)
        results = []
        for comp in comps:
            if comp.sum() < (0.0005 * newW * newH): continue
            quad = mask_to_quad(comp); quad = unclip_quad(quad, unclip, newH, newW)
            score = score_instance(p_map, comp.astype(bool))
            if scale < 1.0: quad = [(x/scale, y/scale) for (x,y) in quad]
            results.append({"polygon": quad, "score": score})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

def draw_polygons(img: Image.Image, results: List[Dict[str,Any]]):
    draw = ImageDraw.Draw(img)
    for r in results:
        poly = r["polygon"]
        if len(poly)==4:
            draw.polygon(poly, outline=(255,0,0))
    return img

def demo_and_save(model_path=None, out_prefix="/mnt/data/dbnet_demo"):
    model = DBNet().to(device)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    det = DBDetector(model, device=device)
    W,H = 768, 512
    img = Image.new("RGB",(W,H),(255,255,255)); d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", 42)
    except Exception:
        font = ImageFont.load_default()
    d.text((40,60),"HELLO WORLD",(0,0,0),font=font)
    d.text((50,140),"This is a DBNet demo",(0,0,0),font=font)
    d.text((60,220),"M1 Max Prototype",(0,0,0),font=font)
    results = det.detect(img, max_side=768, bin_thresh=0.4, unclip=1.6)
    vis = draw_polygons(img.copy(), results)
    img_path = out_prefix + "_result.png"
    json_path = out_prefix + "_result.json"
    vis.save(img_path)
    with open(json_path,"w") as f:
        json.dump({"detections": results}, f, indent=2)
    print("Saved:", img_path, json_path)

if __name__ == "__main__":
    demo_and_save()
