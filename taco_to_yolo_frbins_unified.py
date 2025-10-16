#!/usr/bin/env python3
# taco_to_yolo_frbins_unified.py
# Convertit TACO (COCO) -> YOLOv8 avec mappage en 4 flux français.
# Modes pris en charge :
#   Mode A (ann unique) : --ann et --images-root
#   Mode B (lots/batches): --root avec data/batch_*/annotations.json

import argparse, json, os, shutil, random
from pathlib import Path
from collections import defaultdict, Counter

# ===== 4 flux français (ordre fixe) =====
TARGET_NAMES = ["verre", "emballages", "biodéchets", "résiduel"]
NAME2ID = {n:i for i,n in enumerate(TARGET_NAMES)}

# ===== Mappage TACO -> 4 catégories françaises =====
MERGE = {
    # verre
    "Glass bottle":"verre","Glass jar":"verre","Broken glass":"verre","Other glass":"verre",
    # biodéchets
    "Food waste":"biodéchets",
    # emballages
    "Bottle":"emballages","Bottle cap":"emballages","Metal bottle cap":"emballages","Lid":"emballages",
    "Can":"emballages","Food can":"emballages","Pop tab":"emballages","Aluminum foil":"emballages",
    "Foil wrapper":"emballages","Other metal":"emballages","Metal piece":"emballages",
    "Plastic bottle":"emballages","Plastic lid":"emballages","Plastic straw":"emballages",
    "Plastic wrapper":"emballages","Plastic bag":"emballages","Other plastic bottle":"emballages",
    "Other plastic container":"emballages","Other plastic cup":"emballages","Other plastic lid":"emballages",
    "Other plastic wrapper":"emballages","Bottle label":"emballages",
    "Paper bag":"emballages","Paper carton":"emballages","Paper cup":"emballages","Paper piece":"emballages",
    "Other paper":"emballages","Cardboard":"emballages","Carton":"emballages","Tetra pak":"emballages",
    "Takeaway container":"emballages",
    # résiduel
    "Blister pack":"résiduel","Cigarette":"résiduel","Cup":"résiduel","Disposable plastic cup":"résiduel",
    "Rope":"résiduel","Shoe":"résiduel","Sleeve":"résiduel","Sponge":"résiduel",
    "Styrofoam piece":"résiduel","Styrofoam cup":"résiduel","Other styrofoam":"résiduel",
    "Tissue":"résiduel","Toothbrush":"résiduel","Tube":"résiduel","Wrapper":"résiduel",
    "Other plastic":"résiduel","Other plastic piece":"résiduel","Other plastic straw":"résiduel",
    "Other trash":"résiduel",
}
DEFAULT_CLASS = "résiduel"

# ---------- Utilitaires communs ----------

def map_class(raw_name: str) -> int:
    # Retourne l’ID de la classe cible à partir du nom TACO
    tgt = MERGE.get(raw_name, DEFAULT_CLASS)
    return NAME2ID[tgt]

def load_coco(ann_path: Path):
    # Charge COCO et indexe images, catégories et annotations
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    id2img = {im["id"]: im for im in data["images"]}
    id2cat = {c["id"]: c["name"] for c in data["categories"]}
    imid2anns = defaultdict(list)
    for a in data["annotations"]:
        imid2anns[a["image_id"]].append(a)
    return id2img, id2cat, imid2anns

def ensure_dir(p: Path): 
    # Crée le dossier s’il n’existe pas
    p.mkdir(parents=True, exist_ok=True)

def write_data_yaml(out_root: Path):
    # Écrit data.yaml pour Ultralytics
    yml = (
        f"train: {str(out_root / 'train' / 'images')}\n"
        f"val: {str(out_root / 'val' / 'images')}\n"
        f"nc: {len(TARGET_NAMES)}\n"
        f"names: [{', '.join(TARGET_NAMES)}]\n"
    )
    (out_root / "data.yaml").write_text(yml, encoding="utf-8")

def clamp01(x: float) -> float:
    # Contraint la valeur dans [0,1]
    return 0.0 if x < 0 else 1.0 if x > 1 else x

# ---------- Mode A : ann unique + dossier d’images ----------

def index_images(images_root: Path, exts={".jpg",".jpeg",".png",".bmp",".webp"}):
    # Indexe images : {basename -> chemin} et {stem -> [chemins]}
    by_base, by_stem = {}, defaultdict(list)
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            by_base[p.name] = p
            by_stem[p.stem].append(p)
    return by_base, by_stem

def find_by_filename(file_name_field: str, by_base, by_stem):
    # file_name peut être URL/relatif ; on tente basename puis stem
    base = Path(file_name_field).name
    if base in by_base:
        return by_base[base]
    stem = Path(base).stem
    cand = by_stem.get(stem, [])
    return cand[0] if cand else None

def convert_mode_A(ann: Path, images_root: Path, out_root: Path, val_ratio: float, copy_mode: str, verbose: bool):
    # Convertit à partir d’un ann unique et d’un répertoire d’images déjà téléchargées
    id2img, id2cat, imid2anns = load_coco(ann)
    if verbose: print(f"[A] COCO: images={len(id2img)}, anns≈{sum(len(v) for v in imid2anns.values())}, cats={len(id2cat)}")

    catid2target = {cid: map_class(cname) for cid, cname in id2cat.items()}
    by_base, by_stem = index_images(images_root)
    if verbose: print(f"[A] Index images: {len(by_base)} fichiers sous {images_root}")

    # Appariement image locale
    items, missing = [], 0
    for im_id, im in id2img.items():
        local = find_by_filename(im.get("file_name",""), by_base, by_stem)
        if local is None:
            missing += 1
            continue
        items.append((im_id, im, local))
    if verbose: print(f"[A] Matched={len(items)}, missing={missing}")

    # Split train/val
    random.shuffle(items)
    val_n = int(len(items) * val_ratio)
    val_ids = set(x[0] for x in items[:val_n])

    for sp in ["train","val"]:
        ensure_dir(out_root/sp/"images")
        ensure_dir(out_root/sp/"labels")

    stats = {"train": Counter(), "val": Counter()}

    for im_id, im, src in items:
        sp = "val" if im_id in val_ids else "train"
        dst_img = out_root/sp/"images"/src.name
        if copy_mode == "link":
            try: os.link(src, dst_img)
            except OSError: shutil.copy2(src, dst_img)
        else:
            shutil.copy2(src, dst_img)

        W, H = im.get("width"), im.get("height")
        if not W or not H:
            try:
                from PIL import Image
                with Image.open(src) as imobj:
                    W, H = imobj.width, imobj.height
            except Exception:
                W = H = None

        lines = []
        for a in imid2anns.get(im_id, []):
            if a.get("iscrowd",0)==1 or "bbox" not in a or not W or not H: 
                continue
            x,y,w,h = a["bbox"]
            if w<=0 or h<=0 or W<=0 or H<=0: 
                continue
            cx = clamp01((x + w/2)/W); cy = clamp01((y + h/2)/H)
            ww = clamp01(w/W); hh = clamp01(h/H)
            cls = catid2target.get(a["category_id"], NAME2ID[DEFAULT_CLASS])
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
            stats[sp][cls] += 1

        (out_root/sp/"labels"/(src.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")

    write_data_yaml(out_root)
    if verbose:
        for sp in ["train","val"]:
            per = {TARGET_NAMES[k]: v for k, v in sorted(stats[sp].items())}
            ni = len(list((out_root/sp/"images").glob("*")))
            nl = len(list((out_root/sp/"labels").glob("*.txt")))
            print(f"[A:{sp}] images={ni} labels={nl} per_class={per}")
        print(f"[A] data.yaml -> {out_root/'data.yaml'}")

# ---------- Mode B : lots/batches locaux ----------

def parse_batches(spec: str):
    # Parse "1-10,12,15" -> liste d’entiers sans doublon
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part: continue
        if "-" in part:
            a,b = part.split("-")
            out.extend(range(int(a), int(b)+1))
        else:
            out.append(int(part))
    return sorted(set(out))

def find_image_under_batch(batch_dir: Path, file_name_field: str) -> Path | None:
    # Résout l’image dans un lot : chemin relatif direct, puis batch/<nom>, batch/images/<nom>, puis rglob
    fn = Path(file_name_field)
    cand = batch_dir / fn
    if cand.exists(): return cand
    for sub in [batch_dir, batch_dir/"images"]:
        cand = sub / fn.name
        if cand.exists(): return cand
    hits = list(batch_dir.rglob(fn.name))
    return hits[0] if hits else None

def convert_split_batches(root: Path, out_root: Path, batches: list[int], split: str, copy_mode: str, verbose: bool):
    out_img = out_root/split/"images"
    out_lbl = out_root/split/"labels"
    ensure_dir(out_img); ensure_dir(out_lbl)

    stats_cls = Counter()
    n_img = n_lbl = n_missing_img = n_annjson = 0

    for bi in batches:
        batch_dir = root/"data"/f"batch_{bi}"
        ann = batch_dir/"annotations.json"
        if not ann.exists():
            if verbose: print(f"[B:warn] manquant: {ann}")
            continue
        n_annjson += 1

        id2img, id2cat, imid2anns = load_coco(ann)
        catid2target = {cid: map_class(cname) for cid, cname in id2cat.items()}

        for im_id, im in id2img.items():
            src = find_image_under_batch(batch_dir, im.get("file_name",""))
            if src is None:
                n_missing_img += 1
                continue

            dst_img = out_img/src.name
            if copy_mode == "link":
                try: os.link(src, dst_img)
                except OSError: shutil.copy2(src, dst_img)
            else:
                shutil.copy2(src, dst_img)
            n_img += 1

            W, H = im.get("width"), im.get("height")
            if not W or not H:
                try:
                    from PIL import Image
                    with Image.open(src) as imobj:
                        W, H = imobj.width, imobj.height
                except Exception:
                    W = H = None

            lines = []
            for a in imid2anns.get(im_id, []):
                if a.get("iscrowd",0)==1 or "bbox" not in a or not W or not H:
                    continue
                x,y,w,h = a["bbox"]
                if w<=0 or h<=0 or W<=0 or H<=0:
                    continue
                cx = clamp01((x + w/2)/W); cy = clamp01((y + h/2)/H)
                ww = clamp01(w/W); hh = clamp01(h/H)
                cls = catid2target.get(a["category_id"], NAME2ID[DEFAULT_CLASS])
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
                stats_cls[cls] += 1

            (out_lbl/(src.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            n_lbl += 1

    if verbose:
        per = {TARGET_NAMES[k]: v for k, v in sorted(stats_cls.items())}
        print(f"[B:{split}] images={n_img} labels={n_lbl} ann_files={n_annjson} images_introuvables={n_missing_img}")
    # data.yaml écrit dans main() après train/val

# ---------- Entrée CLI ----------

def main():
    ap = argparse.ArgumentParser("TACO -> YOLOv8 (4 flux FR) | Mode A: --ann + --images-root | Mode B: --root + lots")
    # Mode A
    ap.add_argument("--ann", help="Chemin vers un annotations.json unique (COCO)")
    ap.add_argument("--images-root", help="Racine locale contenant toutes les images téléchargées (recherche récursive)")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Ratio validation si Mode A")
    # Mode B
    ap.add_argument("--root", help="Racine contenant data/batch_*/annotations.json")
    ap.add_argument("--train", default="1-12", help="Lots d’entraînement (ex: 1-12 ou 1-10,12,15)")
    ap.add_argument("--val",   default="13-15", help="Lots de validation")
    # Commun
    ap.add_argument("--out", required=True, help="Dossier de sortie (dataset YOLO)")
    ap.add_argument("--copy-mode", choices=["copy","link"], default="copy", help="Copier les images ou créer des liens durs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    out_root = Path(args.out).resolve()

    # Détection du mode
    mode_A = args.ann and args.images_root
    mode_B = args.root is not None

    if mode_A and mode_B:
        raise SystemExit("Erreur: choisissez soit Mode A (--ann + --images-root), soit Mode B (--root). Pas les deux.")
    if not mode_A and not mode_B:
        raise SystemExit("Erreur: paramètres insuffisants. Mode A: --ann + --images-root. Mode B: --root.")

    if mode_A:
        ann = Path(args.ann).resolve()
        images_root = Path(args.images_root).resolve()
        for sp in ["train","val"]:
            ensure_dir(out_root/sp/"images"); ensure_dir(out_root/sp/"labels")
        convert_mode_A(ann, images_root, out_root, args.val_ratio, args.copy_mode, args.verbose)
        # data.yaml déjà écrit dans convert_mode_A

    else:
        root = Path(args.root).resolve()
        train_batches = parse_batches(args.train)
        val_batches   = parse_batches(args.val)
        for sp in ["train","val"]:
            ensure_dir(out_root/sp/"images"); ensure_dir(out_root/sp/"labels")
        convert_split_batches(root, out_root, train_batches, "train", args.copy_mode, args.verbose)
        convert_split_batches(root, out_root, val_batches,   "val",   args.copy_mode, args.verbose)
        write_data_yaml(out_root)
        if args.verbose:
            print(f"[B] data.yaml -> {out_root/'data.yaml'}")

if __name__ == "__main__":
    main()
