#!/usr/bin/env python3
import os, xml.etree.ElementTree as ET, shutil, cv2
from tqdm import tqdm

# ---- INPUT / OUTPUT PATHS ----
SRC_ROOT = os.path.expanduser("~/Downloads/Manga109_released_2023_12_07")
IMG_ROOT = os.path.join(SRC_ROOT, "images")
ANN_ROOT = os.path.join(SRC_ROOT, "annotations")

OUT_ROOT = "datasets/manga109_dbnet"
OUT_IMG = os.path.join(OUT_ROOT, "images")
OUT_ANN = os.path.join(OUT_ROOT, "annotations")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_ANN, exist_ok=True)

# ---- CONVERSION ----
def convert_one(book_name: str):
    xml_path = os.path.join(ANN_ROOT, f"{book_name}.xml")
    if not os.path.exists(xml_path):
        return
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find all pages
    for page in root.find("pages").findall("page"):
        page_idx = int(page.attrib["index"])
        img_name = f"{page_idx:03d}.jpg"
        img_src = os.path.join(IMG_ROOT, book_name, img_name)
        if not os.path.exists(img_src):
            continue

        # Copy image into flat structure with BookName_000.jpg
        out_img_name = f"{book_name}_{page_idx:03d}.jpg"
        out_img_path = os.path.join(OUT_IMG, out_img_name)
        shutil.copyfile(img_src, out_img_path)

        # Create annotation .txt
        lines = []
        for text in page.findall("text"):
            xmin = int(text.attrib["xmin"])
            ymin = int(text.attrib["ymin"])
            xmax = int(text.attrib["xmax"])
            ymax = int(text.attrib["ymax"])
            # Make a quad polygon (clockwise)
            coords = [
                f"{xmin},{ymin}",
                f"{xmax},{ymin}",
                f"{xmax},{ymax}",
                f"{xmin},{ymax}"
            ]
            lines.append(",".join(coords))
        if lines:
            out_ann_path = os.path.join(OUT_ANN, f"{book_name}_{page_idx:03d}.txt")
            with open(out_ann_path, "w") as f:
                f.write("\n".join(lines))

# ---- MAIN ----
def main():
    books = [os.path.splitext(f)[0] for f in os.listdir(ANN_ROOT) if f.endswith(".xml")]
    print(f"Found {len(books)} books. Starting conversion...")
    for book in tqdm(books):
        convert_one(book)
    print("✅ Done. Output in datasets/manga109_dbnet/")

if __name__ == "__main__":
    main()
