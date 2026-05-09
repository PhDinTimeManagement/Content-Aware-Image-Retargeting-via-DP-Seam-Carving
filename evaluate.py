from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    r = img[..., 0].astype(np.float32)
    g = img[..., 1].astype(np.float32)
    b = img[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def energy_map(img_rgb: np.ndarray) -> np.ndarray:
    gray = rgb_to_gray(img_rgb)
    g = np.pad(gray, pad_width=((1, 1), (1, 1)), mode="edge")
    dx = np.abs(g[1:-1, 2:] - g[1:-1, :-2])
    dy = np.abs(g[2:, 1:-1] - g[:-2, 1:-1])
    return (dx + dy).astype(np.float32)


def avg_energy(img_rgb: np.ndarray) -> float:
    return float(energy_map(img_rgb).mean())


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_rgb(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def baseline_resize(img_rgb: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    im = Image.fromarray(img_rgb, mode="RGB")
    out = im.resize((target_w, target_h), resample=Image.BICUBIC)
    return np.array(out, dtype=np.uint8)


def baseline_center_crop(img_rgb: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    H, W, _ = img_rgb.shape
    if target_w > W or target_h > H:
        raise ValueError("Center crop baseline supports reduction only (target <= current size).")
    left = (W - target_w) // 2
    top = (H - target_h) // 2
    return img_rgb[top:top + target_h, left:left + target_w, :].copy()


@dataclass
class SeamRunResult:
    out_img: np.ndarray
    total_cost: float
    runtime_s: float
    mode: str


def seam_carve(
    img_rgb: np.ndarray,
    target_w: int,
    target_h: int,
    *,
    seam_order: str = "optimal",
    verbose: bool = False,
) -> SeamRunResult:

    t0 = time.time()

    import importlib
    sc = importlib.import_module("seam_carving_dp")

    H, W, _ = img_rgb.shape
    if target_w > W or target_h > H:
        raise ValueError("Seam carving supports reduction only (target <= current size).")

    total_cost = 0.0
    mode_used = seam_order

    # Two-dimension reduction case
    if target_w < W and target_h < H:
        if seam_order == "optimal" and hasattr(sc, "retarget_optimal_order"):
            out, cost = sc.retarget_optimal_order(
                img_rgb,
                target_w=target_w,
                target_h=target_h,
                verbose=verbose,
                print_sequence=False,
            )
            total_cost += float(cost)
            mode_used = "optimal"
        else:
            if seam_order == "height-first":
                out, cost_h = sc.carve_height(img_rgb, target_h=target_h, verbose=verbose)
                total_cost += float(cost_h)
                out, cost_w = sc.carve_width(out, target_w=target_w, verbose=verbose)
                total_cost += float(cost_w)
                mode_used = "height-first"
            else:
                out, cost_w = sc.carve_width(img_rgb, target_w=target_w, verbose=verbose)
                total_cost += float(cost_w)
                out, cost_h = sc.carve_height(out, target_h=target_h, verbose=verbose)
                total_cost += float(cost_h)
                mode_used = "width-first"

    elif target_w < W and target_h == H:
        out, cost_w = sc.carve_width(img_rgb, target_w=target_w, verbose=verbose)
        total_cost += float(cost_w)
        mode_used = "width-only"

    elif target_h < H and target_w == W:
        out, cost_h = sc.carve_height(img_rgb, target_h=target_h, verbose=verbose)
        total_cost += float(cost_h)
        mode_used = "height-only"

    else:
        out = img_rgb.copy()
        mode_used = "none"

    dt = time.time() - t0
    return SeamRunResult(out_img=out, total_cost=total_cost, runtime_s=dt, mode=mode_used)


def list_images(images_root: Path, split: str) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    p = images_root / split
    if not p.exists():
        raise FileNotFoundError(f"Split folder not found: {p}")
    files = [x for x in sorted(p.iterdir()) if x.is_file() and x.suffix.lower() in exts]
    return files


def compute_target_size(
    H: int,
    W: int,
    target_w: Optional[int],
    target_h: Optional[int],
    scale: Optional[float],
) -> Tuple[int, int]:
    tw = target_w if target_w is not None else W
    th = target_h if target_h is not None else H

    if (target_w is None and target_h is None) and (scale is not None):
        tw = max(1, int(round(W * float(scale))))
        th = max(1, int(round(H * float(scale))))

    return int(tw), int(th)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate seam carving using average-energy preservation (paper-style).")
    p.add_argument("--images-root", type=str, default="data/images", help="Root folder containing train/val/test image folders.")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Which split to evaluate.")
    p.add_argument("--methods", type=str, default="seam,resize,crop", help="Comma-separated: seam,resize,crop")
    p.add_argument("--target-width", type=int, default=None, help="Target width (reduction only).")
    p.add_argument("--target-height", type=int, default=None, help="Target height (reduction only).")
    p.add_argument("--scale", type=float, default=None, help="Scale factor for both dimensions if target sizes not provided.")
    p.add_argument("--seam-order", type=str, default="optimal", choices=["optimal", "width-first", "height-first"],
                   help="Order for two-dimension seam carving.")
    p.add_argument("--max-images", type=int, default=None, help="Limit number of images (for quick testing).")
    p.add_argument("--out-csv", type=str, default="output/eval.csv", help="Output CSV file path.")
    p.add_argument("--save-outputs", type=str, default=None, help="Optional folder to save output images for visual inspection.")
    p.add_argument("--no-verbose", action="store_true", help="Disable verbose logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.no_verbose

    images_root = Path(args.images_root)
    split = args.split
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    allowed = {"seam", "resize", "crop"}
    for m in methods:
        if m not in allowed:
            raise ValueError(f"Unknown method '{m}'. Allowed: {sorted(allowed)}")

    img_paths = list_images(images_root, split)
    if args.max_images is not None:
        img_paths = img_paths[: int(args.max_images)]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    save_dir = Path(args.save_outputs) if args.save_outputs is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    if verbose:
        print(f"[info] split={split}, images={len(img_paths)}, methods={methods}")

    for idx, p in enumerate(img_paths, start=1):
        img = load_rgb(p)
        H, W, _ = img.shape

        target_w, target_h = compute_target_size(H, W, args.target_width, args.target_height, args.scale)

        if target_w > W or target_h > H:
            if verbose:
                print(f"[skip] {p.name}: target {target_w}x{target_h} > original {W}x{H} (reduction-only).")
            continue

        e_in = avg_energy(img)

        for method in methods:
            t0 = time.time()

            seam_cost = ""
            seam_mode = ""

            if method == "seam":
                res = seam_carve(img, target_w, target_h, seam_order=args.seam_order, verbose=False)
                out = res.out_img
                runtime = res.runtime_s
                seam_cost = f"{res.total_cost:.6f}"
                seam_mode = res.mode
            elif method == "resize":
                out = baseline_resize(img, target_w, target_h)
                runtime = time.time() - t0
            elif method == "crop":
                out = baseline_center_crop(img, target_w, target_h)
                runtime = time.time() - t0
            else:
                raise RuntimeError("Unreachable method branch.")

            e_out = avg_energy(out)
            delta = e_out - e_in
            ratio = (e_out / e_in) if e_in > 1e-12 else float("inf")

            out_path_str = ""
            if save_dir is not None:
                out_name = f"{p.stem}__{method}__{target_w}x{target_h}.jpg"
                out_path = save_dir / out_name
                save_rgb(out, out_path)
                out_path_str = str(out_path)

            rows.append({
                "image": p.name,
                "split": split,
                "orig_w": W,
                "orig_h": H,
                "target_w": target_w,
                "target_h": target_h,
                "method": method,
                "avgE_in": f"{e_in:.6f}",
                "avgE_out": f"{e_out:.6f}",
                "delta": f"{delta:.6f}",
                "ratio": f"{ratio:.6f}",
                "runtime_s": f"{runtime:.4f}",
                "seam_cost": seam_cost,
                "seam_mode": seam_mode,
                "output_path": out_path_str,
            })

        if verbose and (idx == 1 or idx % 10 == 0 or idx == len(img_paths)):
            print(f"[progress] {idx}/{len(img_paths)} processed: {p.name}")

    fieldnames = [
        "image", "split", "orig_w", "orig_h", "target_w", "target_h", "method",
        "avgE_in", "avgE_out", "delta", "ratio", "runtime_s", "seam_cost", "seam_mode", "output_path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if rows:
        by_method: Dict[str, List[float]] = {}
        by_method_ratio: Dict[str, List[float]] = {}
        for r in rows:
            m = str(r["method"])
            by_method.setdefault(m, []).append(float(r["delta"]))
            by_method_ratio.setdefault(m, []).append(float(r["ratio"]))

        print("\nSummary (higher delta/ratio means more energy preserved under this metric):")
        for m in methods:
            ds = by_method.get(m, [])
            rs = by_method_ratio.get(m, [])
            if not ds:
                continue
            print(f"  {m:6s} | mean(delta)={np.mean(ds):.6f} | mean(ratio)={np.mean(rs):.6f} | n={len(ds)}")

    print(f"\nDone. Wrote CSV to: {out_csv}")
    if save_dir is not None:
        print(f"Saved outputs to: {save_dir}")


if __name__ == "__main__":
    main()
