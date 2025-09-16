#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DBNet smoke test training (Step 1: text detection)
Trains on polygon annotations in datasets/<name>/{images,annotations}

- Expects .jpg images and matching .txt annotation files:
  x1,y1,x2,y2,... per line (polygon vertices)
- Outputs:
    outputs/dbnet_smoke.pth
    outputs/logs/train_log.csv
"""

import os, math, random, csv
from dataclasses import dataclass
from typing import List
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm

from dbnet_text_detector import DBNet, DBLoss   # <- your model/loss file

# ------------------------
# Utils
# ------------------------

def seed_all(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def write_csv_row(path, header, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        w.writerow(row)

def read_polys_txt(path: str) -> List[np.ndarray]:
    polys = []
    if not os.path.exists(path): return polys
    with open(path, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
            if len(parts) < 6: continue
            try:
                nums = list(map(float, parts))
                pts = np.array(nums, dtype=np.int32).reshape(-1,2)
                if len(pts) >= 3:
                    polys.append(pts)
            except: continue
    return polys

def polygons_to_mask(polys: List[np.ndarray], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h,w), dtype=np.uint8)
    for p in polys:
        cv2.fillPoly(mask, [p], 1)
    return mask

def resize_keep_aspect(img, mask, target_long=960):
    h,w = img.shape[:2]
    scale = target_long / max(h,w)
    nh,nw = int(h*scale), int(w*scale)
    img2 = cv2.resize(img, (nw,nh))
    mask2 = cv2.resize(mask, (nw,nh), interpolation=cv2.INTER_NEAREST)
    return img2, mask2

def pad_to_square(img, mask, size=960):
    h,w = img.shape[:2]
    s = max(h,w)
    bg = np.zeros((s,s,3), dtype=np.uint8)
    m2 = np.zeros((s,s), dtype=np.uint8)
    y0 = (s-h)//2; x0 = (s-w)//2
    bg[y0:y0+h,x0:x0+w] = img
    m2[y0:y0+h,x0:x0+w] = mask
    img = cv2.resize(bg, (size,size))
    mask = cv2.resize(m2, (size,size), interpolation=cv2.INTER_NEAREST)
    return img, mask

# ------------------------
# Dataset
# ------------------------

class PolyDataset(Dataset):
    def __init__(self, root, img_size=960, limit=None, val_split=0.1, subset="train"):
        self.img_dir = os.path.join(root,"images")
        self.ann_dir = os.path.join(root,"annotations")
        names = sorted([os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if f.endswith(".jpg")])
        if limit: names = names[:limit]
        if val_split>0:
            random.shuffle(names)
            n_val = int(len(names)*val_split)
            if subset=="train": names = names[n_val:]
            else: names = names[:n_val]
        self.stems = names
        self.img_size = img_size

    def __len__(self): return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img = cv2.imread(os.path.join(self.img_dir, stem+".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w = img.shape[:2]
        polys = read_polys_txt(os.path.join(self.ann_dir, stem+".txt"))
        mask = polygons_to_mask(polys, h, w)
        img, mask = resize_keep_aspect(img, mask, target_long=1280)
        img, mask = pad_to_square(img, mask, size=self.img_size)
        img = img.transpose(2,0,1).astype(np.float32)/255.0
        mask = mask[None].astype(np.float32)
        return torch.from_numpy(img), torch.from_numpy(mask)

# ------------------------
# Training
# ------------------------

@dataclass
class PhaseCfg:
    name: str
    root: str
    epochs: int
    batch_size: int
    lr: float
    img_size: int
    val_split: float = 0.1
    limit: int = None
    patience: int = 3
    save_path: str = ""

def f1_from_logits(logits: torch.Tensor, mask: torch.Tensor, thresh=0.3):
    with torch.no_grad():
        prob = torch.sigmoid(logits)
        binm = (prob>thresh).float()
        tp = (binm*mask).sum().item()
        fp = (binm*(1-mask)).sum().item()
        fn = ((1-binm)*mask).sum().item()
        prec = tp/(tp+fp+1e-8)
        rec = tp/(tp+fn+1e-8)
        return 2*prec*rec/(prec+rec+1e-8)

class Trainer:
    def __init__(self, device): self.device = device

    def run_phase(self, model, cfg, log_csv):
        ds_tr = PolyDataset(cfg.root, cfg.img_size, cfg.limit, cfg.val_split, "train")
        ds_va = PolyDataset(cfg.root, cfg.img_size, cfg.limit, cfg.val_split, "val")
        dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True)
        dl_va = DataLoader(ds_va, batch_size=cfg.batch_size)

        print(f"Phase {cfg.name}: train={len(ds_tr)} val={len(ds_va)}")
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs*len(dl_tr))
        criterion = DBLoss()
        best_f1, bad = 0.0, 0

        for ep in range(1,cfg.epochs+1):
            model.train()
            pbar = tqdm(dl_tr, desc=f"{cfg.name} epoch {ep}/{cfg.epochs}")
            tr_loss, step = 0,0
            for imgs,masks in pbar:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                P,T,B = model(imgs)
                if masks.shape[-2:] != P.shape[-2:]:
                    masks = torch.nn.functional.interpolate(masks, size=P.shape[-2:], mode="nearest")
                loss,_ = criterion(P,T,B,masks)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tr_loss += loss.item(); step+=1
                if step%10==0:
                    fg = masks.sum().item()/(masks.numel()+1e-8)*100
                    pbar.set_postfix(loss=tr_loss/step, fg=f"{fg:.2f}%")

            model.eval()
            va_loss, va_f1, n = 0,0,0
            with torch.no_grad():
                for imgs,masks in dl_va:
                    imgs,masks = imgs.to(self.device), masks.to(self.device)
                    P,T,B = model(imgs)
                    if masks.shape[-2:] != P.shape[-2:]:
                        masks = torch.nn.functional.interpolate(masks, size=P.shape[-2:], mode="nearest")
                    loss,_ = criterion(P,T,B,masks)
                    va_loss += loss.item()
                    va_f1 += f1_from_logits(P,masks); n+=1
            va_loss/=max(1,n); va_f1/=max(1,n)
            print(f" val loss={va_loss:.4f} f1={va_f1:.4f}")

            write_csv_row(log_csv,["phase","epoch","val_f1"],[cfg.name,ep,f"{va_f1:.4f}"])
            if va_f1>best_f1:
                best_f1=va_f1; torch.save(model.state_dict(), cfg.save_path)
                print("  ✅ saved",cfg.save_path)
                bad=0
            else:
                bad+=1
                if bad>=cfg.patience:
                    print("⏹ early stop"); break
        return cfg.save_path,best_f1

# ------------------------
# Main
# ------------------------

def main():
    seed_all(1337)
    ensure_dir("outputs"); ensure_dir("outputs/logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = DBNet().to(device)
    trainer = Trainer(device)

    # ---- SMOKE TEST CONFIG ----
    cfg = PhaseCfg(
        name="smoke_manga",
        root="datasets/manga109_dbnet",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        img_size=768,
        val_split=0.1,
        limit=240,
        save_path="outputs/dbnet_smoke.pth"
    )
    trainer.run_phase(model, cfg, "outputs/logs/train_log.csv")

if __name__ == "__main__":
    main()
