#!/usr/bin/env python3
# Mode unique : un seul annotations.json + images déjà téléchargées réparties en 15 dossiers batch_1..batch_15
# Conversion COCO -> YOLOv8 avec mappage 4 flux FR + renommage anti-collision (préfixe de lot).

import argparse, json, os, shutil, random, re
from pathlib import Path
from collections import defaultdict, Counter

# ===== 4 flux français (ordre fixe) =====
TARGET_NAMES = ["verre", "emballages", "biodéchets", "résiduel"]
NAME2ID = {n: i for i, n in enumerate(TARGET_NAMES)}

# ===== Mappage TACO -> 4 catégories françaises =====
MERGE = {
    # verre
    "Glass bottle":"verre","Glass jar":"verre","Broken glass":"verre","Other glass":"verre",
    # biodéchets
    "Food waste":"biodéchets",
    # emballages (papiers/cartons/métaux/plastiques d’emballage)
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

# ---------- utilitaires ----------

def map_class(raw_name: str) -> int:
    # Nom TACO -> ID de classe cible
    return NAME2ID[MERGE.get(raw_name, DEFAULT_CLASS)]

def clamp01(x: float) -> float:
    # Contraint dans [0,1]
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def ensure_dir(p: Path):
    # Crée le dossier si absent
    p.mkdir(parents=True, exist_ok=True)

def load_coco(ann_path: Path):
    # Charge COCO et indexe images/annos/catégories
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    id2img = {im["id"]: im for im in data["images"]}
    id2cat = {c["id"]: c["name"] for c in data["categories"]}
    imid2anns = defaultdict(list)
    for a in data["annotations"]:
        imid2anns[a["image_id"]].append(a)
    return id2img, id2cat, imid2anns

def index_images_with_batches(images_root: Path, exts={".jpg",".jpeg",".png",".bmp",".webp"}):
    """
    Indexe récursivement les images et retient toutes les occurrences d’un même basename.
    Retourne:
      - by_base: {basename -> [chemins possibles]}
      - batch_of: {chemin -> 'b{num}' ou 'b0' si non détecté}
    """
    by_base = defaultdict(list)
    batch_of = {}
    pat = re.compile(r"batch[_-](\d+)", re.IGNORECASE)
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            by_base[p.name].append(p)
            m = pat.search(str(p.parent))
            batch_of[p] = f"b{m.group(1)}" if m else "b0"
    return by_base, batch_of

def resolve_local_image(file_name_field: str, by_base: dict, used_paths: set) -> Path | None:
    """
    Résout une image locale pour un file_name (URL possible).
    S’il y a plusieurs candidats portant le même basename, choisit celui non encore utilisé.
    À défaut, retourne le premier.
    """
    base = Path(file_name_field).name
    cands = by_base.get(base, [])
    if not cands:
        return None
    for p in cands:
        if p not in used_paths:
            return p
    return cands[0]

def write_data_yaml(out_root: Path):
    # data.yaml relatif au dossier courant (évite préfixes dupliqués)
    (out_root / "data.yaml").write_text(
        "train: train/images\n"
        "val: val/images\n"
        f"nc: {len(TARGET_NAMES)}\n"
        f"names: [{', '.join(TARGET_NAMES)}]\n",
        encoding="utf-8"
    )

# ---------- conversion (Mode A + renommage anti-collision) ----------

def convert_mode_A_collision_safe(
    ann: Path,
    images_root: Path,
    out_root: Path,
    val_ratio: float,
    copy_mode: str,
    seed: int,
    verbose: bool
):
    random.seed(seed)

    # Charge COCO
    id2img, id2cat, imid2anns = load_coco(ann)
    catid2target = {cid: map_class(cname) for cid, cname in id2cat.items()}
    if verbose:
        print(f"[A] images={len(id2img)}  anns≈{sum(len(v) for v in imid2anns.values())}  cats={len(id2cat)}")

    # Indexe toutes les images disponibles + repère le batch d’origine
    by_base, batch_of = index_images_with_batches(images_root)
    if verbose:
        total_files = sum(len(v) for v in by_base.values())
        print(f"[A] index local: {total_files} fichiers sous {images_root}")

    # Apparier chaque image COCO à un fichier local
    items, missing = [], 0
    used_paths = set()
    for im_id, im in id2img.items():
        local = resolve_local_image(im.get("file_name", ""), by_base, used_paths)
        if local is None:
            missing += 1
            continue
        used_paths.add(local)
        items.append((im_id, im, local))
    if verbose:
        print(f"[A] appariés={len(items)}  manquants={missing}")

    # Split train/val
    random.shuffle(items)
    val_n = int(len(items) * val_ratio)
    val_ids = set(x[0] for x in items[:val_n])

    # Dossiers de sortie
    for sp in ["train", "val"]:
        ensure_dir(out_root / sp / "images")
        ensure_dir(out_root / sp / "labels")

    # Conversion + renommage: b{batch}_{basename}
    stats = {"train": Counter(), "val": Counter()}
    for im_id, im, src in items:
        sp = "val" if im_id in val_ids else "train"
        prefix = batch_of.get(src, "b0")
        dst_name = f"{prefix}_{src.name}"
        dst_img = out_root / sp / "images" / dst_name

        if copy_mode == "link":
            try:
                os.link(src, dst_img)
            except OSError:
                shutil.copy2(src, dst_img)
        else:
            shutil.copy2(src, dst_img)

        # Dimensions image
        W, H = im.get("width"), im.get("height")
        if not W or not H:
            try:
                from PIL import Image
                with Image.open(src) as imobj:
                    W, H = imobj.width, imobj.height
            except Exception:
                W = H = None

        # Étiquette YOLO
        lines = []
        for a in imid2anns.get(im_id, []):
            if a.get("iscrowd", 0) == 1 or "bbox" not in a or not W or not H:
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0 or W <= 0 or H <= 0:
                continue
            cx = clamp01((x + w / 2) / W)
            cy = clamp01((y + h / 2) / H)
            ww = clamp01(w / W)
            hh = clamp01(h / H)
            cls = catid2target.get(a["category_id"], NAME2ID[DEFAULT_CLASS])
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
            stats[sp][cls] += 1

        (out_root / sp / "labels" / (Path(dst_name).stem + ".txt")).write_text(
            "\n".join(lines), encoding="utf-8"
        )

    write_data_yaml(out_root)
    if verbose:
        for sp in ["train", "val"]:
            per = {TARGET_NAMES[k]: v for k, v in sorted(stats[sp].items())}
            ni = len(list((out_root / sp / "images").glob("*")))
            nl = len(list((out_root / sp / "labels").glob("*.txt")))
            print(f"[A:{sp}] images={ni}  labels={nl}  per_class={per}")
        print(f"[A] data.yaml -> {out_root/'data.yaml'}")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        "Mode A unique : --ann + --images-root | COCO -> YOLOv8 4 flux FR, renommage anti-collision par lot"
    )
    ap.add_argument("--ann", required=True, help="Chemin vers un annotations.json unique (COCO)")
    ap.add_argument("--images-root", required=True, help="Racine locale contenant toutes les images (batch_1..batch_15)")
    ap.add_argument("--out", required=True, help="Dossier de sortie du dataset YOLO")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Ratio de validation (ex: 0.2)")
    ap.add_argument("--copy-mode", choices=["copy", "link"], default="copy", help="Copier les images ou créer des liens durs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    convert_mode_A_collision_safe(
        ann=Path(args.ann).resolve(),
        images_root=Path(args.images_root).resolve(),
        out_root=Path(args.out).resolve(),
        val_ratio=args.val_ratio,
        copy_mode=args.copy_mode,
        seed=args.seed,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
