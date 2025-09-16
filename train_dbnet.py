#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DBNet training — Step 1: Text Detection
Two-phase training: SynthText pretrain → Manga109 finetune

Dataset structure expected:
datasets/synthtext_dbnet/images/*.jpg
datasets/synthtext_dbnet/annotations/*.txt
datasets/manga109_dbnet/images/*.jpg
datasets/manga109_dbnet/annotations/*.txt

Each .txt has polygon coords: x1,y1,x2,y2,...
"""

import os, math, random, csv
from dataclasses import dataclass
import cv2, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw

from dbnet_text_detector import DBNet, DBLoss  # your model+loss

# --------------------- Utilities ---------------------

def seed_all(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def write_csv_row(path, header, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        w.writerow(row)

def polygon_txt_to_mask(txt_path, h, w):
    mask = Image.new("L",(w,h),0)
    if not os.path.exists(txt_path): return np.zeros((h,w),np.uint8)
    with open(txt_path,"r") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            pts = list(map(int,line.split(",")))
            xy = [(pts[i],pts[i+1]) for i in range(0,len(pts),2)]
            ImageDraw.Draw(mask).polygon(xy, outline=1, fill=1)
    return np.array(mask,np.uint8)

def tensorize(img, mask):
    img = img.transpose(2,0,1).astype(np.float32)/255.0
    mask = mask[None].astype(np.float32)
    return torch.from_numpy(img), torch.from_numpy(mask)

# --------------------- Dataset ---------------------

class DBNetFlatDir(Dataset):
    def __init__(self, root, img_size=960, augment=False, val_split=0.0, limit=None, subset='train'):
        self.img_dir = os.path.join(root,"images")
        self.ann_dir = os.path.join(root,"annotations")
        names = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg",".png",".jpeg"))]
        if limit: names = names[:limit]
        if val_split>0:
            random.shuffle(names)
            n_val = int(len(names)*val_split)
            if subset=='train': names = names[n_val:]
            else: names = names[:n_val]
        self.stems = names
        self.img_size = img_size
        self.augment = augment

    def __len__(self): return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img = cv2.imread(os.path.join(self.img_dir, stem+".jpg"))
        if img is None:
            img = cv2.imread(os.path.join(self.img_dir, stem+".png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w = img.shape[:2]
        mask = polygon_txt_to_mask(os.path.join(self.ann_dir, stem+".txt"), h,w)
        img = cv2.resize(img,(self.img_size,self.img_size))
        mask = cv2.resize(mask,(self.img_size,self.img_size),interpolation=cv2.INTER_NEAREST)
        img,mask = tensorize(img,mask)
        return img,mask

# --------------------- Training ---------------------

@dataclass
class PhaseCfg:
    name: str; root: str; epochs: int; batch_size: int; lr: float; img_size: int
    val_split: float=0.02; limit: int|None=None; save_path: str=""

class Trainer:
    def __init__(self, device):
        self.device = device

    def run_phase(self, model, cfg: PhaseCfg):
        ds_tr = DBNetFlatDir(cfg.root,cfg.img_size,augment=True,val_split=cfg.val_split,limit=cfg.limit,subset='train')
        ds_va = DBNetFlatDir(cfg.root,cfg.img_size,augment=False,val_split=cfg.val_split,limit=cfg.limit,subset='val')
        dl_tr = DataLoader(ds_tr,batch_size=cfg.batch_size,shuffle=True,num_workers=2,drop_last=True)
        dl_va = DataLoader(ds_va,batch_size=cfg.batch_size,shuffle=False,num_workers=2)

        print(f"Phase [{cfg.name}] — train {len(ds_tr)} / val {len(ds_va)} images")
        bce = nn.BCELoss()
        opt = torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=1e-4)

        best_f1 = 0
        for ep in range(1,cfg.epochs+1):
            model.train(); pbar=tqdm(dl_tr,desc=f"{cfg.name} | epoch {ep}/{cfg.epochs}")
            for imgs,masks in pbar:
                imgs,masks = imgs.to(self.device), masks.to(self.device)
                P,T,B = model(imgs)
                if masks.shape[-2:]!=P.shape[-2:]:
                    masks = torch.nn.functional.interpolate(masks,size=P.shape[-2:],mode="nearest")
                loss = bce(P.squeeze(1), masks.squeeze(1))
                opt.zero_grad(); loss.backward(); opt.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # --- val ---
            model.eval(); tot_f1=0; n=0
            with torch.no_grad():
                for imgs,masks in dl_va:
                    imgs,masks = imgs.to(self.device), masks.to(self.device)
                    P,T,B = model(imgs)
                    if masks.shape[-2:]!=P.shape[-2:]:
                        masks = torch.nn.functional.interpolate(masks,size=P.shape[-2:],mode="nearest")
                    prob = torch.sigmoid(P)
                    binm = (prob>0.3).float()
                    tp = (binm*masks).sum().item()
                    fp = (binm*(1-masks)).sum().item()
                    fn = ((1-binm)*masks).sum().item()
                    prec = tp/(tp+fp+1e-8); rec = tp/(tp+fn+1e-8)
                    f1 = 2*prec*rec/(prec+rec+1e-8)
                    tot_f1+=f1; n+=1
            mean_f1 = tot_f1/max(1,n)
            print(f"  val F1: {mean_f1:.4f}")
            if mean_f1>best_f1:
                best_f1=mean_f1
                torch.save(model.state_dict(), cfg.save_path)
                print(f"  ✅ saved {cfg.save_path}")

        return cfg.save_path,best_f1

# --------------------- Main ---------------------

def main():
    seed_all()
    ensure_dir("outputs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = DBNet().to(device)
    trainer = Trainer(device)

    USE_SMOKE = True  # ⬅️ set to False for full training

    if USE_SMOKE:
        synth_cfg = PhaseCfg("pretrain_synthtext_smoke","datasets/synthtext_dbnet",1,8,1e-3,640,val_split=0.01,limit=200,save_path="outputs/dbnet_pre_smoke.pth")
        manga_cfg = PhaseCfg("finetune_manga109_smoke","datasets/manga109_dbnet",1,4,5e-4,768,val_split=0.02,limit=100,save_path="outputs/dbnet_trained_smoke.pth")
    else:
        synth_cfg = PhaseCfg("pretrain_synthtext","datasets/synthtext_dbnet",15,6,1e-3,960,val_split=0.01,save_path="outputs/dbnet_pretrained.pth")
        manga_cfg = PhaseCfg("finetune_manga109","datasets/manga109_dbnet",20,4,5e-4,1024,val_split=0.02,save_path="outputs/dbnet_trained.pth")

    best_pre,_ = trainer.run_phase(model,synth_cfg)
    model.load_state_dict(torch.load(best_pre,map_location=device))
    best_ft,_ = trainer.run_phase(model,manga_cfg)
    print("✅ Done. Final weights:", best_ft)

if __name__=="__main__":
    main()
