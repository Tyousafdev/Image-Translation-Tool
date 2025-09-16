#!/usr/bin/env python3
import os, math, random, time, json
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ----------------- MODEL -----------------
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
        return c1,c2,c3,c4

class FPNNeck(nn.Module):
    def __init__(self, in_channels=(32,64,128,256), out_c=128):
        super().__init__()
        self.lat4 = nn.Conv2d(in_channels[3], out_c, 1)
        self.lat3 = nn.Conv2d(in_channels[2], out_c, 1)
        self.lat2 = nn.Conv2d(in_channels[1], out_c, 1)
        self.lat1 = nn.Conv2d(in_channels[0], out_c, 1)
        self.smooth3 = ConvBNReLU(out_c, out_c)
        self.smooth2 = ConvBNReLU(out_c, out_c)
        self.smooth1 = ConvBNReLU(out_c, out_c)
    def forward(self, c1,c2,c3,c4):
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode='nearest')
        p3 = self.smooth3(p3); p2 = self.smooth2(p2); p1 = self.smooth1(p1)
        return p1,p2,p3,p4

class DBHead(nn.Module):
    def __init__(self, in_c=128, k=50):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(ConvBNReLU(in_c*4, in_c), ConvBNReLU(in_c, in_c//2))
        self.p_out = nn.Conv2d(in_c//2, 1, 1)
        self.t_out = nn.Conv2d(in_c//2, 1, 1)
    def forward(self, p1,p2,p3,p4):
        size = p1.shape[-2:]
        p2u = F.interpolate(p2, size=size, mode='bilinear', align_corners=False)
        p3u = F.interpolate(p3, size=size, mode='bilinear', align_corners=False)
        p4u = F.interpolate(p4, size=size, mode='bilinear', align_corners=False)
        x = torch.cat([p1,p2u,p3u,p4u], dim=1)
        x = self.conv(x)
        P = torch.sigmoid(self.p_out(x))
        T = torch.sigmoid(self.t_out(x))
        B = torch.sigmoid(self.k * (P - T))
        return P,T,B

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

# ----------------- DATASET -----------------
class MangaDataset(Dataset):
    def __init__(self, root, img_size=1024, limit=None):
        self.img_paths = sorted(glob(os.path.join(root, "images", "*.jpg")) +
                                glob(os.path.join(root, "images", "*.png")))
        if limit: self.img_paths = self.img_paths[:limit]
        self.mask_paths = [p.replace("images","masks").rsplit(".",1)[0]+".png" for p in self.img_paths]
        self.tf = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor()
        ])
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, i):
        img = Image.open(self.img_paths[i]).convert("RGB")
        mask = Image.open(self.mask_paths[i]).convert("L")
        return self.tf(img), self.tf(mask)

# ----------------- TRAINING -----------------
@dataclass
class PhaseCfg:
    name: str
    root: str
    epochs: int
    batch_size: int
    lr: float
    img_size: int
    val_split: float
    limit: int = None
    save_path: str = "outputs/dbnet_trained.pth"

class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        if torch.cuda.is_available():
            self.autocast = torch.autocast(device_type='cuda', dtype=torch.float16)
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            self.autocast = torch.cpu.amp.autocast()
            self.scaler = torch.amp.GradScaler("cpu", enabled=False)

    def run_phase(self, cfg: PhaseCfg):
        all_ds = MangaDataset(cfg.root, cfg.img_size, limit=cfg.limit)
        n_val = max(1, int(len(all_ds)*cfg.val_split))
        ds_train, ds_val = torch.utils.data.random_split(all_ds, [len(all_ds)-n_val, n_val])

        dl_tr = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=4, pin_memory=True, drop_last=True)
        dl_va = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        best_f1 = 0
        for ep in range(cfg.epochs):
            self.model.train()
            for imgs, masks in dl_tr:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                with self.autocast:
                    P,T,B = self.model(imgs)
                    loss_p = self.criterion_bce(P, masks)
                    loss_t = self.criterion_l1(T, masks)
                    loss_b = self.criterion_bce(B, masks)
                    loss = loss_p + 0.5*loss_t + 0.5*loss_b
                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
                opt.zero_grad(set_to_none=True)

            # val
            self.model.eval()
            tp=fp=fn=0; val_loss=0
            with torch.no_grad():
                for imgs,masks in dl_va:
                    imgs,masks = imgs.to(self.device), masks.to(self.device)
                    with self.autocast:
                        P,_,_ = self.model(imgs)
                        preds = (torch.sigmoid(P)>0.5).float()
                        val_loss += self.criterion_bce(P,masks).item()
                    tp += ((preds*masks)>0).sum().item()
                    fp += ((preds*(1-masks))>0).sum().item()
                    fn += (((1-preds)*masks)>0).sum().item()
            precision = tp/(tp+fp+1e-6); recall = tp/(tp+fn+1e-6)
            f1 = 2*precision*recall/(precision+recall+1e-6)
            print(f"[{cfg.name}] epoch {ep+1}/{cfg.epochs} loss={val_loss/len(dl_va):.4f} f1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                torch.save(self.model.state_dict(), cfg.save_path)
                print(f"  ✅ new best f1={f1:.4f} saved {cfg.save_path}")

# ----------------- MAIN -----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DBNet()
    trainer = Trainer(model, device)

    os.makedirs("outputs", exist_ok=True)

    synth_cfg = PhaseCfg(
        name="pretrain_synthtext",
        root="datasets/synthtext_dbnet",
        epochs=25, batch_size=16, lr=1e-3, img_size=1024,
        val_split=0.01, limit=15000, save_path="outputs/dbnet_pretrained.pth"
    )
    manga_cfg = PhaseCfg(
        name="finetune_manga109",
        root="datasets/manga109_dbnet",
        epochs=40, batch_size=12, lr=5e-4, img_size=1280,
        val_split=0.02, save_path="outputs/dbnet_trained.pth"
    )

    trainer.run_phase(synth_cfg)
    trainer.run_phase(manga_cfg)

    print("🎉 Training complete — final weights:", manga_cfg.save_path)
