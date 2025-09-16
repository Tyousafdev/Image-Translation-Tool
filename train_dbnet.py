#!/usr/bin/env python3
import os, math, random
from dataclasses import dataclass
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dbnet_text_detector import DBNet

# -----------------------
# Dataset
# -----------------------

class DBNetDataset(Dataset):
    def __init__(self, root, img_size=640, augment=False, limit=None, val_split=0.0, subset='train'):
        self.img_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "annotations")
        names = sorted([os.path.splitext(f)[0] for f in os.listdir(self.img_dir)
                        if f.lower().endswith((".jpg",".jpeg",".png"))])
        if limit: names = names[:limit]
        if val_split > 0:
            random.shuffle(names)
            n_val = int(len(names) * val_split)
            if subset == 'train':
                names = names[n_val:]
            else:
                names = names[:n_val]
        self.names = names
        self.img_size = img_size
        self.augment = augment

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.img_dir, name + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, name + ".png")
        ann_path = os.path.join(self.ann_dir, name + ".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        mask = Image.new("L", (w, h), 0)

        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                for line in f:
                    pts = [int(float(x)) for x in line.strip().split(",")]
                    if len(pts) >= 6:
                        xy = [(pts[i], pts[i+1]) for i in range(0, len(pts), 2)]
                        ImageDraw.Draw(mask).polygon(xy, outline=1, fill=1)

        img = img.resize((self.img_size, self.img_size))
        mask = mask.resize((self.img_size, self.img_size))

        img = torch.from_numpy(np.array(img).transpose(2,0,1)).float()/255.0
        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0)
        return img, mask

# -----------------------
# Training
# -----------------------

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
        self.bce = nn.BCEWithLogitsLoss()

    def run_phase(self, model, cfg: PhaseCfg):
        ds_train = DBNetDataset(cfg.root, cfg.img_size, augment=True, limit=cfg.limit, val_split=cfg.val_split, subset='train')
        ds_val   = DBNetDataset(cfg.root, cfg.img_size, augment=False, limit=cfg.limit, val_split=cfg.val_split, subset='val')
        dl_tr = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
        dl_va = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

        print(f"Phase [{cfg.name}] — train {len(ds_train)} / val {len(ds_val)} images")
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        best_f1 = -1

        for ep in range(1, cfg.epochs+1):
            model.train()
            pbar = tqdm(dl_tr, desc=f"{cfg.name} | epoch {ep}/{cfg.epochs}")
            for imgs, masks in pbar:
                imgs = imgs.to(self.device)
                masks = masks.float().to(self.device)
                if masks.max() > 1:
                    masks = masks / 255.0
                masks = masks.clamp(0,1)

                P, T, B = model(imgs)
                if masks.shape[-2:] != P.shape[-2:]:
                    masks = torch.nn.functional.interpolate(masks, size=P.shape[-2:], mode="nearest")

                loss = self.bce(P.squeeze(1), masks.squeeze(1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                pbar.set_postfix(loss=loss.item())

            # validation
            model.eval()
            with torch.no_grad():
                total_f1 = 0; n = 0
                for imgs, masks in dl_va:
                    imgs = imgs.to(self.device)
                    masks = masks.float().to(self.device)
                    if masks.max() > 1:
                        masks = masks / 255.0
                    masks = masks.clamp(0,1)

                    P, T, B = model(imgs)
                    if masks.shape[-2:] != P.shape[-2:]:
                        masks = torch.nn.functional.interpolate(masks, size=P.shape[-2:], mode="nearest")

                    prob = torch.sigmoid(P)
                    pred = (prob > 0.3).float()
                    tp = (pred*masks).sum().item()
                    fp = (pred*(1-masks)).sum().item()
                    fn = ((1-pred)*masks).sum().item()
                    prec = tp/(tp+fp+1e-8); rec = tp/(tp+fn+1e-8)
                    f1 = 2*prec*rec/(prec+rec+1e-8)
                    total_f1 += f1; n += 1
                avg_f1 = total_f1 / max(1,n)
                print(f"val F1: {avg_f1:.4f}")
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    torch.save(model.state_dict(), cfg.save_path)
                    print(f"  ✅ saved {cfg.save_path}")

        return cfg.save_path, best_f1

# -----------------------
# Main
# -----------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = DBNet().to(device)
    trainer = Trainer(device)

    USE_SMOKE = True
    if USE_SMOKE:
        synth_cfg = PhaseCfg(
            name="pretrain_synthtext_smoke",
            root="datasets/synthtext_dbnet",
            epochs=1, batch_size=8, lr=1e-3, img_size=640,
            val_split=0.01, limit=200, save_path="outputs/dbnet_pretrained_smoke.pth"
        )
        manga_cfg = PhaseCfg(
            name="finetune_manga109_smoke",
            root="datasets/manga109_dbnet",
            epochs=1, batch_size=4, lr=5e-4, img_size=640,
            val_split=0.02, limit=100, save_path="outputs/dbnet_trained_smoke.pth"
        )
    else:
        synth_cfg = PhaseCfg(
            name="pretrain_synthtext",
            root="datasets/synthtext_dbnet",
            epochs=15, batch_size=6, lr=1e-3, img_size=960,
            val_split=0.01, save_path="outputs/dbnet_pretrained.pth"
        )
        manga_cfg = PhaseCfg(
            name="finetune_manga109",
            root="datasets/manga109_dbnet",
            epochs=20, batch_size=4, lr=5e-4, img_size=1024,
            val_split=0.02, save_path="outputs/dbnet_trained.pth"
        )

    best_pre,_ = trainer.run_phase(model, synth_cfg)
    model.load_state_dict(torch.load(best_pre, map_location=device))
    best_ft,_ = trainer.run_phase(model, manga_cfg)
    print("✅ Done. Final:", best_ft)

if __name__ == "__main__":
    main()
