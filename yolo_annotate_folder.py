#!/usr/bin/env python
"""
yolo_annotate_folder.py
-----------------------

Annotate every image in <images dir> with YOLO predictions and save
the overlays into <out dir>, replicating the folder structure.

Requirements
------------
pip install ultralytics opencv-python tqdm
"""
import argparse, sys
from pathlib import Path

import cv2
from ultralytics import YOLO
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def collect_images(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            yield p

def draw_boxes(img, boxes, names, hide_labels):
    for b in boxes:
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if not hide_labels:
            label = f"{names[cls]} {conf:.2f}"
            tsize, _ = cv2.getTextSize(label,
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            lbl_x2 = x1 + tsize[0] + 4
            lbl_y2 = y1 - tsize[1] - 4
            cv2.rectangle(img, (x1, y1),
                               (lbl_x2, lbl_y2), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="YOLO .pt checkpoint")
    ap.add_argument("--images", required=True,
                    help="Folder containing images (searched recursively)")
    ap.add_argument("--out", required=True,
                    help="Target folder for annotated images")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold")
    ap.add_argument("--hide-labels", action="store_true",
                    help="Suppress class/conf text, keep only boxes")
    args = ap.parse_args()

    in_root  = Path(args.images).expanduser()
    out_root = Path(args.out).expanduser()
    if not in_root.is_dir():
        sys.exit(f"❌  images path {in_root} is not a directory")

    out_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    names = model.names

    imgs = list(collect_images(in_root))
    if not imgs:
        sys.exit("❌  No images found")

    for p in tqdm(imgs, desc="annotating"):
        img = cv2.imread(str(p))
        if img is None:
            print("⚠️  could not read", p); continue

        res = model(img, conf=args.conf, verbose=False)[0]
        img = draw_boxes(img, res.boxes, names, args.hide_labels)

        rel = p.relative_to(in_root)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)

    print("✓ Done. Annotated images saved to", out_root)

if __name__ == "__main__":
    main()
