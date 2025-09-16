#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-phase DBNet training (SynthText pretrain -> Manga109 finetune) for Step 1: text detection.

Outputs:
  outputs/dbnet_pretrained.pth   (best SynthText phase)
  outputs/dbnet_trained.pth      (best Manga109 phase)
  outputs/logs/*.csv             (loss / f1 logs)
"""

import os, math, random, csv
from dataclasses import dataclass
import cv2, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- Your model/loss (from your earlier Step-1 file) ---
from dbnet_text_detector import DBNet, DBLoss

# ------------------------
# Utilities
# ------------------------

def seed_all(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def write_csv_row(path, header, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        w.writerow(row)

def polygon_txt_to_mask(txt_path: str, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if not os.path.exists(txt_path): 
        return mask
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            nums = list(map(int, line.split(",")))
            pts = np.array(nums, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
    return mask

def resize_keep_aspect(img, target_long=1024):
    h, w = img.shape[:2]
    scale = target_long / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img2 = cv2.resize(img, (nw, nh))
    return img2, scale

def random_augment(img, mask):
    h, w = img.shape[:2]
    s = 0.6 + random.random()*1.0
    img = cv2.resize(img, (int(w*s), int(h*s)))
    mask = cv2.resize(mask, (int(w*s), int(h*s)), interpolation=cv2.INTER_NEAREST)

    angle = random.uniform(-7, 7)
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if random.random() < 0.5:
        img = cv2.flip(img, 1); mask = cv2.flip(mask, 1)
    if random.random() < 0.1:
        img = cv2.flip(img, 0); mask = cv2.flip(mask, 0)

    if random.random() < 0.8:
        alpha = 0.85 + random.random()*0.3
        beta  = random.uniform(-12, 12)
        img = np.clip(img.astype(np.float32)*alpha + beta, 0, 255).astype(np.uint8)

    if random.random() < 0.3:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(35, 70)]
        _, enc = cv2.imencode(".jpg", img, encode_param)
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return img, mask

def pad_to_square(img, mask, size=960):
    h, w = img.shape[:2]
    s = max(h, w)
    bg = np.zeros((s, s, 3), dtype=np.uint8)
    m2 = np.zeros((s, s), dtype=np.uint8)
    y0 = (s - h) // 2; x0 = (s - w) // 2
    bg[y0:y0+h, x0:x0+w] = img
    m2[y0:y0+h, x0:x0+w] = mask
    img = cv2.resize(bg, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(m2, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, mask

def tensorize(img, mask):
    img = img.transpose(2,0,1).astype(np.float32) / 255.0
    mask = mask[None].astype(np.float32)
    return torch.from_numpy(img), torch.from_numpy(mask)

# ------------------------
# Datasets
# ------------------------

class DBNetFlatDir(Dataset):
    def __init__(self, root, img_size=960, augment=False, limit=None, val_split=0.0, subset='train'):
        self.img_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "annotations")
        names = sorted([os.path.splitext(f)[0] for f in os.listdir(self.img_dir)
                        if f.lower().endswith((".jpg",".jpeg",".png"))])
        if limit: names = names[:limit]
        if val_split > 0:
            n = len(names)
            n_val = int(round(n*val_split))
            random.shuffle(names)
            if subset == 'train': names = names[n_val:]
            else: names = names[:n_val]
        self.stems = names
        self.img_size = img_size
        self.augment = augment

    def __len__(self): return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img_path = None
        for ext in (".jpg",".jpeg",".png"):
            p = os.path.join(self.img_dir, stem+ext)
            if os.path.exists(p): img_path = p; break
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mask = polygon_txt_to_mask(os.path.join(self.ann_dir, stem+".txt"), h, w)
        if self.augment:
            img, mask = random_augment(img, mask)
        img, _ = resize_keep_aspect(img, target_long=1280)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        img, mask = pad_to_square(img, mask, size=self.img_size)
        img, mask = tensorize(img, mask)
        return {"image": img, "mask": mask}

# ------------------------
# Training / Validation
# ------------------------

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
    patience: int = 5
    save_path: str = ""

def f1_from_logits(P: torch.Tensor, mask: torch.Tensor, thresh=0.3) -> float:
    with torch.no_grad():
        prob = torch.sigmoid(P)
        binm = (prob > thresh).float()
        tp = (binm * mask).sum().item()
        fp = (binm * (1-mask)).sum().item()
        fn = ((1-binm) * mask).sum().item()
        prec = tp / (tp+fp+1e-8)
        rec  = tp / (tp+fn+1e-8)
        return 2*prec*rec/(prec+rec+1e-8)

class Trainer:
    def __init__(self, device):
        self.device = device
        if torch.backends.mps.is_available():
    # 🔧 Disable autocast on MPS (causes dtype mismatch errors)
            self.autocast_ctx = torch.autocast(device_type='mps', enabled=False)
        elif torch.cuda.is_available():
            self.autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        else:
            self.autocast_ctx = torch.autocast(device_type='cpu', dtype=torch.bfloat16)

        # Only enable GradScaler on CUDA (not needed on MPS)
        self.scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available())


    def run_phase(self, model: nn.Module, cfg: PhaseCfg, log_csv: str):
        ds_train = DBNetFlatDir(cfg.root, img_size=cfg.img_size, augment=True, limit=cfg.limit, val_split=cfg.val_split, subset='train')
        ds_val   = DBNetFlatDir(cfg.root, img_size=cfg.img_size, augment=False, limit=cfg.limit, val_split=cfg.val_split, subset='val')
        dl_tr = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)
        dl_va = DataLoader(ds_val,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)

        print(f"Phase [{cfg.name}] — train {len(ds_train)} / val {len(ds_val)} images")

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4, betas=(0.9, 0.999))
        total_steps = max(1, cfg.epochs * len(dl_tr))
        warmup_steps = min(1500, int(0.03 * total_steps))
        def lr_lambda(step):
            if step < warmup_steps:
                return max(1e-3, step / max(1, warmup_steps))
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * t))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        criterion = DBLoss()
        best_f1 = -1.0
        bad = 0

        for ep in range(1, cfg.epochs+1):
            model.train()
            pbar = tqdm(dl_tr, desc=f"{cfg.name} | epoch {ep}/{cfg.epochs}")
            tr_loss = 0.0; step = 0

            for batch in pbar:
                imgs = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                with self.autocast_ctx:
                    P, T, B = model(imgs)
                    if masks.shape[-2:] != P.shape[-2:]:
                        masks = torch.nn.functional.interpolate(masks, size=P.shape[-2:], mode="nearest")
                    loss, _ = criterion(P, T, B, masks)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step(); sched.step()

                tr_loss += loss.item(); step += 1
                if step % 10 == 0:
                    pbar.set_postfix(loss=f"{tr_loss/step:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

            # --- validate ---
            model.eval()
            va_loss = 0.0; va_f1 = 0.0; n = 0
            with torch.no_grad():
                for batch in dl_va:
                    imgs = batch["image"].to(self.device)
                    masks = batch["mask"].to(self.device)
                    P, T, B = model(imgs)
                    if masks.shape[-2:] != P.shape[-2:]:
                        masks = torch.nn.functional.interpolate(masks, size=P.shape[-2:], mode="nearest")
                    loss, _ = criterion(P, T, B, masks)
                    va_loss += loss.item()
                    va_f1   += f1_from_logits(P, masks)
                    n += 1
            va_loss /= max(1,n); va_f1 /= max(1,n)
            print(f"  val: loss={va_loss:.4f}  pixel-F1={va_f1:.4f}")

            write_csv_row(log_csv, ["phase","epoch","train_loss","val_loss","val_f1","lr"],
                          [cfg.name, ep, f"{tr_loss/max(1,step):.6f}", f"{va_loss:.6f}", f"{va_f1:.6f}", f"{sched.get_last_lr()[0]:.3e}"])

            if va_f1 > best_f1:
                best_f1 = va_f1
                torch.save(model.state_dict(), cfg.save_path)
                print(f"  ✅ new best {best_f1:.4f} -> saved {cfg.save_path}")
                bad = 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    print("  ⏹ early stop (no F1 improvement)")
                    break

        return cfg.save_path, best_f1

# ------------------------
# Main
# ------------------------

def main():
    seed_all(1337)
    ensure_dir("outputs"); ensure_dir("outputs/logs")

    if not torch.cuda.is_available():
        raise RuntimeError("❌ No GPU detected! Please enable GPU in Colab: Runtime > Change runtime type > GPU")

    device = torch.device("cuda")
    print("✅ Using GPU:", torch.cuda.get_device_name(0))

    print("Device:", device)

    model = DBNet().to(device)

    synth_cfg = PhaseCfg(
        name="pretrain_synthtext",
        root="datasets/synthtext_dbnet",
        epochs=15,
        batch_size=6,
        lr=1e-3,
        img_size=960,
        val_split=0.01,
        limit=5000,
        save_path="outputs/dbnet_pretrained.pth"
    )
    trainer = Trainer(device)
    log_csv = "outputs/logs/train_log.csv"
    best_pre_ckpt, best_pre_f1 = trainer.run_phase(model, synth_cfg, log_csv)
    print(f"Pretraining done. Best pixel-F1={best_pre_f1:.4f}")

    sd = torch.load(best_pre_ckpt, map_location=device)
    model.load_state_dict(sd, strict=True)

    manga_cfg = PhaseCfg(
        name="finetune_manga109",
        root="datasets/manga109_dbnet",
        epochs=20,
        batch_size=4,
        lr=5e-4,
        img_size=1024,
        val_split=0.02,
        save_path="outputs/dbnet_trained.pth"
    )
    best_ft_ckpt, best_ft_f1 = trainer.run_phase(model, manga_cfg, log_csv)
    print(f"Finetune done. Best pixel-F1={best_ft_f1:.4f}")
    print("✅ Final weights:", best_ft_ckpt)

if __name__ == "__main__":
    main()
