import os, random, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
from pathlib import Path

from dbnet_text_detector import DBNet  # your DBNet model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
# Dataset Loader
# ============================
class TextDataset(Dataset):
    def __init__(self, root, img_size=1024, limit=None, transform=None):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.ann_dir = self.root / "annotations"
        self.transform = transform
        self.img_size = img_size

        self.samples = sorted(self.img_dir.glob("*.jpg"))
        if limit: self.samples = self.samples[:limit]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        ann_path = self.ann_dir / (img_path.stem + ".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Create binary mask from polygons
        mask = Image.new("L", (w, h), 0)
        if ann_path.exists():
            with open(ann_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    coords = line.replace(',', ' ').split()
                    # make pairs: [x1,y1,x2,y2,...] -> [(x1,y1), (x2,y2)...]
                    try:
                        pts = [(float(coords[i]), float(coords[i+1])) for i in range(0, len(coords), 2)]
                    except ValueError:
                        continue
                    if len(pts) >= 3:
                        ImageDraw.Draw(mask).polygon(pts, outline=1, fill=1)


        if self.transform:
            seed = random.randint(0, 2**32)
            random.seed(seed); torch.manual_seed(seed)
            img = self.transform(img)
            random.seed(seed); torch.manual_seed(seed)
            mask = self.transform(mask)

        return img, mask


# ============================
# Config Struct
# ============================
class PhaseCfg:
    def __init__(self, name, root, epochs, batch_size, lr, img_size, val_split, save_path, limit=None):
        self.name = name
        self.root = root
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.img_size = img_size
        self.val_split = val_split
        self.save_path = save_path
        self.limit = limit


# ============================
# Trainer
# ============================
class Trainer:
    def __init__(self, device):
        self.device = device

    def run_phase(self, model, cfg: PhaseCfg, log_csv="outputs/logs/train_log.csv"):
        tfm = transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor()
        ])
        dataset = TextDataset(cfg.root, cfg.img_size, limit=cfg.limit, transform=tfm)
        val_len = max(1, int(len(dataset) * cfg.val_split))
        train_len = len(dataset) - val_len
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])

        dl_tr = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
        dl_val = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

        model = model.to(self.device)
        opt = optim.AdamW(model.parameters(), lr=cfg.lr)
        bce = nn.BCEWithLogitsLoss()

        best_f1 = 0.0
        os.makedirs(Path(cfg.save_path).parent, exist_ok=True)

        for epoch in range(cfg.epochs):
            model.train()
            for imgs, masks in dl_tr:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                preds = model(imgs)
                loss = bce(preds.squeeze(1), masks.squeeze(1))
                opt.zero_grad(); loss.backward(); opt.step()

            # ---------- validation ----------
            model.eval()
            tp = fp = fn = 0
            val_loss = 0
            with torch.no_grad():
                for imgs, masks in dl_val:
                    imgs, masks = imgs.to(self.device), masks.to(self.device)
                    preds = torch.sigmoid(model(imgs))
                    val_loss += bce(preds.squeeze(1), masks.squeeze(1)).item()
                    bin_preds = (preds > 0.3).float()
                    tp += (bin_preds * masks).sum().item()
                    fp += (bin_preds * (1 - masks)).sum().item()
                    fn += ((1 - bin_preds) * masks).sum().item()
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)

            print(f"[{cfg.name}] epoch {epoch+1}/{cfg.epochs} "
                  f"val_loss={val_loss/len(dl_val):.4f} f1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), cfg.save_path)
                print(f"✅ new best {best_f1:.4f} -> saved {cfg.save_path}")

        return cfg.save_path, best_f1


# ============================
# Main
# ============================
if __name__ == "__main__":
    os.makedirs("outputs/logs", exist_ok=True)

    model = DBNet()

    trainer = Trainer(device)

    # --------------------------
    # ⚡ Choose smoke or full
    # --------------------------
    USE_SMOKE = True   # ⬅️ flip this to True for quick ~20min run

    if USE_SMOKE:
        synth_cfg = PhaseCfg(
            name="pretrain_synthtext_smoke",
            root="datasets/synthtext_dbnet",
            epochs=1, batch_size=8, lr=1e-3, img_size=768,
            val_split=0.01, limit=1000, save_path="outputs/dbnet_pretrained_smoke.pth"
        )
        manga_cfg = PhaseCfg(
            name="finetune_manga109_smoke",
            root="datasets/manga109_dbnet",
            epochs=2, batch_size=4, lr=5e-4, img_size=896,
            val_split=0.02, limit=200, save_path="outputs/dbnet_trained_smoke.pth"
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

    # --------------------------
    # 🔁 Run training
    # --------------------------
    best_pre_ckpt, best_pre_f1 = trainer.run_phase(model, synth_cfg)
    model.load_state_dict(torch.load(best_pre_ckpt, map_location=device), strict=True)
    best_final_ckpt, best_final_f1 = trainer.run_phase(model, manga_cfg)

    print(f"\n✅ Training complete. Best F1: {best_final_f1:.4f}")
    print(f"Model saved at: {best_final_ckpt}")
