#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DBNet training — Step 1: text detection
Dataset structure:
datasets/
  synthtext_dbnet/
    images/*.jpg
    annotations/*.txt
  manga109_dbnet/
    images/*.jpg
    annotations/*.txt
"""

import os, random, math, csv
from pathlib import Path
from dataclasses import dataclass
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from dbnet_text_detector import DBNet  # your model

# ------------------
# Utilities
# ------------------

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); 
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def write_csv(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        w.writerow(row)

# ------------------
# Dataset
# ------------------

class DBNetFlatDir(Dataset):
    def __init__(self, root, img_size=960, augment=False, limit=None, val_split=0.0, subset='train'):
        self.img_dir = Path(root) / "images"
        self.ann_dir = Path(root) / "annotations"
        stems = [p.stem for p in self.img_dir.glob("*.*")]
        if limit: stems = stems[:limit]
        if val_split > 0:
            random.shuffle(stems)
            n_val = int(len(stems)*val_split)
            if subset == 'val': stems = stems[:n_val]
            else: stems = stems[n_val:]
        self.stems = stems
        self.img_size = img_size
        self.augment = augment
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img_path = next(self.img_dir.glob(stem + ".*"))
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        mask = Image.new("L", (w, h), 0)
        ann_path = self.ann_dir / (stem + ".txt")
        if ann_path.exists():
            with open(ann_path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    coords = line.replace(',', ' ').split()
                    try:
                        pts = [(float(coords[i]), float(coords[i+1])) for i in range(0, len(coords), 2)]
                    except (ValueError, IndexError):
                        continue
                    if len(pts) >= 3:
                        ImageDraw.Draw(mask).polygon(pts, outline=1, fill=1)
        img = self.tf(img)
        mask = transforms.Resize((self.img_size,self.img_size))(mask)
        mask = transforms.ToTensor()(mask)
        return img, mask

# ------------------
# Trainer
# ------------------

@dataclass
class PhaseCfg:
    name: str
    root: str
    epochs: int
    batch_size: int
    lr: float
    img_size: int
    val_split: float = 0.02
    limit: int = None
    save_path: str = ""

class Trainer:
    def __init__(self, device):
        self.device = device

    def run_phase(self, model, cfg: PhaseCfg):
        ds_tr = DBNetFlatDir(cfg.root, img_size=cfg.img_size, augment=True, limit=cfg.limit, val_split=cfg.val_split, subset='train')
        ds_va = DBNetFlatDir(cfg.root, img_size=cfg.img_size, augment=False, limit=cfg.limit, val_split=cfg.val_split, subset='val')
        dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

        print(f"Phase [{cfg.name}] — train {len(ds_tr)} / val {len(ds_va)} images")
        x, y = ds_tr[0]
        print(f"[DEBUG] sample shapes: img={x.shape}, mask={y.shape}")

        bce = nn.BCELoss()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

        best_f1 = 0.0
        for ep in range(1, cfg.epochs+1):
            model.train()
            pbar = tqdm(dl_tr, desc=f"{cfg.name} | epoch {ep}/{cfg.epochs}")
            for imgs, masks in pbar:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                out = model(imgs)
                if isinstance(out, (tuple, list)): preds = out[0]
                else: preds = out
                loss = bce(preds.squeeze(1), masks.squeeze(1))
                opt.zero_grad(); loss.backward(); opt.step()
                pbar.set_postfix(loss=loss.item())

            # --- simple val ---
            model.eval(); tot = 0; correct = 0
            with torch.no_grad():
                for imgs, masks in dl_va:
                    imgs, masks = imgs.to(self.device), masks.to(self.device)
                    out = model(imgs)
                    if isinstance(out, (tuple, list)): preds = out[0]
                    else: preds = out
                    binm = (torch.sigmoid(preds) > 0.5).float()
                    correct += (binm*masks).sum().item()
                    tot += masks.sum().item()
            f1 = correct / (tot+1e-6)
            print(f"  val approx-F1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), cfg.save_path)
                print(f"  ✅ saved {cfg.save_path}")
        return cfg.save_path, best_f1

# ------------------
# Main
# ------------------

def main():
    seed_all(1337)
    ensure_dir("outputs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = DBNet().to(device)
    trainer = Trainer(device)

    # Smoke-test configs
    synth_cfg = PhaseCfg(
        name="pretrain_synthtext",
        root="datasets/synthtext_dbnet",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        img_size=640,
        val_split=0.01,
        limit=200,
        save_path="outputs/dbnet_pretrained.pth"
    )
    manga_cfg = PhaseCfg(
        name="finetune_manga109",
        root="datasets/manga109_dbnet",
        epochs=1,
        batch_size=2,
        lr=5e-4,
        img_size=800,
        val_split=0.02,
        limit=200,
        save_path="outputs/dbnet_trained.pth"
    )

    best_pre, _ = trainer.run_phase(model, synth_cfg)
    model.load_state_dict(torch.load(best_pre, map_location=device))
    best_ft, _ = trainer.run_phase(model, manga_cfg)
    print("✅ Finished. Final weights:", best_ft)

if __name__ == "__main__":
    main()
