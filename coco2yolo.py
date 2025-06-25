#!/usr/bin/env python
"""
Convert SSBI COCO annotations to YOLO txt
----------------------------------------
Usage: python coco2yolo.py --json instances_train.json --img-dir images/train --out-dir labels/train
"""

import json, argparse, pathlib, tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--json',     required=True)
parser.add_argument('--img-dir',  required=True)
parser.add_argument('--out-dir',  required=True)
args = parser.parse_args()

img_dir  = pathlib.Path(args.img_dir)
out_dir  = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

coco = json.load(open(args.json))

# 1. COCO helper tables
img_id2name = {im["id"]: im["file_name"] for im in coco["images"]}
cat_id2idx  = {c["id"]: i for i, c in enumerate(sorted(coco["categories"],
                                 key=lambda x: x["id"]))}

# 2. Collect per-image annotations
annos = {}
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    x, y, w, h = ann["bbox"]
    img_w = next(im["width"] for im in coco["images"] if im["id"] == img_id)
    img_h = next(im["height"] for im in coco["images"] if im["id"] == img_id)

    # YOLO format: class x_center y_center width height (all normalised 0-1)
    xc, yc = x + w/2, y + h/2
    line = f'{cat_id2idx[ann["category_id"]]} {xc/img_w:.6f} {yc/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}\n'
    annos.setdefault(img_id, []).append(line)

# 3. Write txt per image
for img_id, lines in annos.items():
    txt_path = out_dir / (pathlib.Path(img_id2name[img_id]).stem + '.txt')
    txt_path.write_text(''.join(lines))

print(f"✓ Wrote {len(annos)} label files to {out_dir}")


