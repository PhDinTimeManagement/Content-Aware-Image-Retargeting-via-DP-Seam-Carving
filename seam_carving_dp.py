from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image


# ==============================
# I/O utilities
# ==============================

# Load image of shape (H, W, 3) in RGB order
def load_image_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")  # Force 3-channel RGB
    return np.array(img, dtype=np.uint8)

# Save image of shape (H, W, 3) in RGB order
def save_image_rgb(arr: np.ndarray, path: Path) -> None:
    """Save (H, W, 3) uint8 RGB image."""
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


# ==============================
# Energy function
# ==============================

# Convert RGB image to grayscale using luminosity method
def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    r = img[..., 0].astype(np.float32)          # Red channel
    g = img[..., 1].astype(np.float32)          # Green channel
    b = img[..., 2].astype(np.float32)          # Blue channel
    return 0.299 * r + 0.587 * g + 0.114 * b    # Luminosity method to grayscale

# Compute energy map E(H, W) using an L1 gradient magnitude approximation
def energy_map(img: np.ndarray) -> np.ndarray:
    gray = rgb_to_gray(img)  # Gradient is easier to compute on grayscale

    # Pad by edge replication (in four directions), so central differences work at borders
    g = np.pad(gray, pad_width=((1, 1), (1, 1)), mode="edge")

    # Central differences (L1)
    dx = np.abs(g[1:-1, 2:] - g[1:-1, :-2])   # Horizontal gradient, right - left
    dy = np.abs(g[2:, 1:-1] - g[:-2, 1:-1])   # Vertical gradient, down - up

    E = dx + dy # Pixels with strong edges/textures have higher energy and are treated as more important
    return E.astype(np.float32)


# ==============================
# Seam Dynamic Programming
# ==============================

@dataclass
class SeamResult:
    seam_cols: np.ndarray  # For vertical seam: length H, seam_cols[i] is column at row i
    seam_cost: float       # Total energy along the seam

# Find the minimum-energy vertical seam via dynamic programming
def find_vertical_seam(E: np.ndarray) -> SeamResult:
    """
        DP table:
            M[i, j] = minimum cumulative energy to reach pixel (i, j) from the top row

        Transition:
            M[i, j] = E[i, j] + min(M[i-1, j-1], M[i-1, j], M[i-1, j+1])

        Initialization:
            M[0, j] = E[0, j] for all columns j

        Backtracking:
            back[i, j] ∈ {-1, 0, +1} indicates which predecessor column was chosen
            (i-1, j-1), (i-1, j), or (i-1, j+1), respectively
    """
    H, W = E.shape
    M = E.copy()
    back = np.zeros((H, W), dtype=np.int16)

    # DP forward pass: fill rows from top to bottom
    for i in range(1, H):   # Start from the 2nd row because row 0 has no predecessor
        prev = M[i - 1]     # prev[j] = cumulative min energy to reach pixel (i-1, j)

        # We want, for each column j, to consider three possible predecessors:
        #         prev[j-1], prev[j], prev[j+1]
        #         To do this vectorized, we build three shifted arrays with inf padding at boundaries

        left = np.pad(prev[:-1], (1, 0), mode="constant", constant_values=np.inf)       # prev[j-1], for j=0, prev[-1] is invalid -> set to +inf
        mid = prev                                                                      # prev[j], directly above
        right = np.pad(prev[1:], (0, 1), mode="constant", constant_values=np.inf)       # prev[j+1], for j=W-1, prev[W] is invalid -> set to +inf

        # Stack them so choices[k, j] is the k-th predecessor cost for column j
        # k=0 -> left-up, k=1 -> up, k=2 -> right-up
        choices = np.stack([left, mid, right], axis=0)  # (3, W)

        # For each column j, pick which predecessor gives the minimum cumulative cost
        argmin = np.argmin(choices, axis=0)            # (W,), values in {0,1,2}

        # Extract the actual minimum predecessor cumulative energy for each column j
        minval = choices[argmin, np.arange(W)]         # (W,), selected predecessor cost

        # Record the best step for backtracking:
        # 0 -> -1 (came from j-1), 1 -> 0 (came from j), 2 -> +1 (came from j+1)
        back[i] = (argmin.astype(np.int16) - 1)

        # Vectorized update for the whole row i
        M[i] = E[i] + minval

    # The seam ends at the minimum cumulative energy position in the last row
    j = int(np.argmin(M[-1]))       # Column index of the seam endpoint at bottom row
    seam_cost = float(M[-1, j])     # Total minimum seam energy (objective value)

    # Backtrack from bottom to top to recover the seam path (one column per row)
    seam_cols = np.zeros(H, dtype=np.int32)     # seam_cols[i] = selected column at row i
    seam_cols[-1] = j                           # Set bottom row column
    for i in range(H - 1, 0, -1):               # Go upward: H-1, H-2, ..., 1
        j = j + int(back[i, j])                 # Move to the predecessor column in row i-1
        j = max(0, min(j, W - 1))               # Clamp for safety (should not go out of bounds)
        seam_cols[i - 1] = j                    # Store seam column for row i-1

    return SeamResult(seam_cols=seam_cols, seam_cost=seam_cost)

# Remove vertical seam from RGB image
def remove_vertical_seam(img: np.ndarray, seam_cols: np.ndarray) -> np.ndarray:
    H, W, C = img.shape
    assert seam_cols.shape == (H,)  # Must provide exactly one seam column index per row

    # Output image has width reduced by 1 because we remove exactly 1 pixel per row
    out = np.empty((H, W - 1, C), dtype=img.dtype)      # Allocate result array

    # Process each row independently (because seam selects one column per row)
    for i in range(H):
        j = int(seam_cols[i])       # Column index of seam pixel to remove in this row i

        # Keep pixels left of j, and pixels right of j, then concatenate them.
        # img[i, :j, :]      -> pixels before seam column
        # img[i, j+1:, :]    -> pixels after seam column
        out[i, :, :] = np.concatenate([img[i, :j, :], img[i, j + 1 :, :]], axis=0)      # axis=0 concatenates along width

    # Finally return image with width W-1
    return out


# ==============================
# Single-dimension carving
# ==============================

# Reduce height to target_h by removing minimum-energy horizontal seams
def carve_height(img: np.ndarray, target_h: int, *, verbose: bool = True) -> Tuple[np.ndarray, float]:
    H, W, _ = img.shape
    if target_h > H:
        raise ValueError(f"target_h={target_h} > current height={H}. This script implements reduction only.")
    if target_h <= 0:
        raise ValueError("target_h must be positive.")

    # Trick: removing a horizontal seam in (H, W) is equivalent to:
    # 1) transpose the image to (W, H)
    # 2) remove a vertical seam there
    # 3) transpose back
    transposed = np.transpose(img, (1, 0, 2))   # (H, W, 3) -> (W, H, 3)

    # Now transposed width equals original height
    # To reduce original height to target_h, reduce transposed width to target_h
    carved_t, total_cost = carve_width(transposed, target_w=target_h, verbose=verbose)

    # Transpose back to restore (H, W, 3) layout
    carved = np.transpose(carved_t, (1, 0, 2))  # (W, target_h, 3) -> (target_h, W, 3)

    # Return height-carved image + total cost
    return carved, total_cost

# Reduce width to target_w by repeatedly removing minimum-energy vertical seams
def carve_width(img: np.ndarray, target_w: int, *, verbose: bool = True) -> Tuple[np.ndarray, float]:
    # Read current image dimensions
    H, W, _ = img.shape

    # Only support reduction
    if target_w > W:
        raise ValueError(f"target_w={target_w} > current width={W}. This script implements reduction only.")
    if target_w <= 0:
        raise ValueError("target_w must be positive.")

    # Accumulate total seam cost removed
    total_cost = 0.0

    # Compute how many vertical seams we need to remove
    seams_to_remove = W - target_w

    # Remove exactly one seam per iteration
    for t in range(seams_to_remove):
        # Recompute energy map on the CURRENT image (it changes after each removal)
        E = energy_map(img)

        # Find the minimum-energy vertical seam via DP
        seam = find_vertical_seam(E)

        # Remove that seam from the image (width decreases by 1)
        img = remove_vertical_seam(img, seam.seam_cols)

        # Add seam cost into total cost
        total_cost += seam.seam_cost

        if verbose and (t == 0 or (t + 1) % 25 == 0 or (t + 1) == seams_to_remove):
            print(f"[width] removed seam {t+1:4d}/{seams_to_remove}, seam_cost={seam.seam_cost:.2f}")

    # Return the carved image and the total cost removed
    return img, total_cost


# ============================================================
# Two-Dimension Retargeting: single seam removal steps
# ============================================================

def remove_one_vertical(img: np.ndarray) -> Tuple[np.ndarray, float]:
    # Compute the energy map for the current image
    E = energy_map(img)

    # Find the minimum-energy vertical seam via DP
    seam = find_vertical_seam(E)

    # Remove that seam from the image
    img2 = remove_vertical_seam(img, seam.seam_cols)

    # Return the updated image and the seam cost for transport-map accumulation
    return img2, seam.seam_cost


def find_horizontal_seam(E: np.ndarray) -> SeamResult:
    # A horizontal seam in E(H,W) is equivalent to a vertical seam in the transposed map E.T(W,H)
    seam_t = find_vertical_seam(E.T)

    # seam_t.seam_cols has length W; it indicates, for each original column, which row to remove
    return SeamResult(seam_cols=seam_t.seam_cols, seam_cost=seam_t.seam_cost)


def remove_horizontal_seam(img: np.ndarray, seam_rows: np.ndarray) -> np.ndarray:
    # Transpose the image so that removing a horizontal seam becomes removing a vertical seam
    img_t = np.transpose(img, (1, 0, 2))        # (H, W, 3) -> (W, H, 3)

    # Remove the seam as a vertical seam in the transposed image (seam_rows length must equal W)
    carved_t = remove_vertical_seam(img_t, seam_rows)

    # Transpose back to return an (H-1, W, 3) image
    return np.transpose(carved_t, (1, 0, 2))    # (W, H-1, 3) -> (H-1, W, 3)


def remove_one_horizontal(img: np.ndarray) -> Tuple[np.ndarray, float]:
    # Compute the energy map for the current image
    E = energy_map(img)

    # Find the minimum-energy horizontal seam (via vertical seam on E.T)
    seam = find_horizontal_seam(E)

    # Remove the horizontal seam from the image
    img2 = remove_horizontal_seam(img, seam.seam_cols)

    # Return the updated image and the seam cost for transport-map accumulation
    return img2, seam.seam_cost


# ============================================================
# Two-Dimension Retargeting: optimal order via transport map
# ============================================================

def retarget_optimal_order(img0: np.ndarray,
                            target_w: int,
                            target_h: int,
                            *,
                            verbose: bool = True,
                            print_sequence: bool = True,) -> Tuple[np.ndarray, float]:

    # Read original image size
    H0, W0, _ = img0.shape

    # Compute how many seams we must remove in each direction
    R = H0 - target_h       # Number of horizontal seams to remove
    C = W0 - target_w       # Number of vertical seams to remove

    # Ensure we only support reduction
    if R < 0 or C < 0:
        raise ValueError("retarget_optimal_order supports reduction only (target <= current size).")

    # Allocate DP cost table: T[r,c] is minimal cumulative cost to reach (H0-r, W0-c)
    T = np.full((R + 1, C + 1), np.inf, dtype=np.float64)

    # Allocate DP choice table: 0=from left (vertical), 1=from top (horizontal)
    choice = np.zeros((R + 1, C + 1), dtype=np.uint8)

    # Base case. No seams removed yet has cost 0
    T[0, 0] = 0.0

    # prev_imgs[c] will store the image for DP state (r-1, c)
    prev_imgs: List[Optional[np.ndarray]] = [None] * (C + 1)

    # curr_imgs[c] will store the image for DP state (r, c)
    curr_imgs: List[Optional[np.ndarray]] = [None] * (C + 1)

    # -------------------------
    # Initialize row r = 0
    # -------------------------

    # State (0,0) image is the original
    curr_imgs[0] = img0

    # Fill first row by repeatedly removing vertical seams
    for c in range(1, C + 1):
        # Remove one vertical seam from state (0, c-1)
        img_new, cost = remove_one_vertical(curr_imgs[c - 1])   # type: ignore[arg-type]

        # Store the resulting image at state (0, c)
        curr_imgs[c] = img_new

        # Update minimal cost for T[0, c]
        T[0, c] = T[0, c - 1] + cost

        # Record that we reached (0, c) from the left (vertical removal)
        choice[0, c] = 0

    # Row 0 becomes the previous row for the next iteration (r = 1)
    prev_imgs = curr_imgs[:]

    # -------------------------
    # Fill rows r = 1..R
    # -------------------------

    for r in range(1, R + 1):
        # Reset current row image cache
        curr_imgs = [None] * (C + 1)

        # First column c=0: only horizontal seam removals are possible
        img_new, cost = remove_one_horizontal(prev_imgs[0])     # type: ignore[arg-type]
        curr_imgs[0] = img_new
        T[r, 0] = T[r - 1, 0] + cost
        choice[r, 0] = 1                # From top

        # General cells. Choose between removing H seam (from top) or V seam (from left)
        for c in range(1, C + 1):
            # Option A. Come from top (r-1, c) then remove one horizontal seam
            imgA, costA = remove_one_horizontal(prev_imgs[c])  # type: ignore[arg-type]
            valA = T[r - 1, c] + costA

            # Option B. Come from left (r, c-1) then remove one vertical seam
            imgB, costB = remove_one_vertical(curr_imgs[c - 1]) # type: ignore[arg-type]
            valB = T[r, c - 1] + costB

            # Pick the cheaper option and store both the chosen cost and resulting image
            if valA <= valB:
                T[r, c] = valA
                choice[r, c] = 1
                curr_imgs[c] = imgA
            else:
                T[r, c] = valB
                choice[r, c] = 0
                curr_imgs[c] = imgB

        # Progress logging per DP row
        if verbose:
            print(f"[optimal] finished DP row {r}/{R}")

        # Move to the next DP row
        prev_imgs = curr_imgs[:]

    # The final image is stored at DP state (R, C)
    out_img = prev_imgs[C]

    # Total minimal cumulative cost is T[R, C]
    total_cost = float(T[R, C])

    # Optionally backtrack the operation sequence (H/V) for debugging or reporting
    if verbose and print_sequence:
        ops: List[str] = []
        rr, cc = R, C
        while rr > 0 or cc > 0:
            if choice[rr, cc] == 1:
                ops.append("H")
                rr -= 1
            else:
                ops.append("V")
                cc -= 1
        ops.reverse()
        print(f"[optimal] op sequence length={len(ops)} (H={R}, V={C})")

    # Return the retargeted image and total cumulative cost
    return out_img, total_cost


# ============================================================
# CLI and main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seam carving (DP) for content-aware image reduction.")
    p.add_argument("--input", required=True, type=str, help="Path to input image (jpg/png).")
    p.add_argument("--output", required=True, type=str, help="Path to save output image.")
    p.add_argument("--target-width", type=int, default=None, help="Target width in pixels (reduction only).")
    p.add_argument("--target-height", type=int, default=None, help="Target height in pixels (reduction only).")
    p.add_argument("--no-verbose", action="store_true", help="Disable progress printing.")
    p.add_argument("--no-sequence", action="store_true", help="Do not print H/V operation sequence.")
    return p.parse_args()

def main() -> None:
    # Parse command-line arguments and set up I/O paths
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    verbose = not args.no_verbose

    # Load input image and get original dimensions
    img = load_image_rgb(in_path)
    H, W, _ = img.shape

    # Determine target dimensions
    target_w = args.target_width if args.target_width is not None else W
    target_h = args.target_height if args.target_height is not None else H

    # Validate reduction-only constraints
    if target_w > W or target_h > H:
        raise ValueError("This implementation supports reduction only (target <= current size).")

    # Start timing and accumulate total seam-removal cost
    t0 = time.time()
    total_cost = 0.0

    # Case 1. Reduce BOTH width and height -> use optimal order retargeting
    if target_w < W and target_h < H:
        img, cost_opt = retarget_optimal_order(
            img,
            target_w=target_w,
            target_h=target_h,
            verbose=verbose,
            print_sequence=(not args.no_sequence),
        )
        total_cost += cost_opt

    # Case 2. Reduce ONLY width -> use repeated vertical seam removal
    elif target_w < W and target_h == H:
        img, cost_w = carve_width(img, target_w=target_w, verbose=verbose)
        total_cost += cost_w

    # Case 3. Reduce ONLY height -> use repeated horizontal seam removal
    elif target_h < H and target_w == W:
        img, cost_h = carve_height(img, target_h=target_h, verbose=verbose)
        total_cost += cost_h

    # Case 4. No resizing requested -> just copy input to output
    else:
        if verbose:
            print("[info] target size equals original size; no seams removed.")

    # Save the output image and print summary
    dt = time.time() - t0
    save_image_rgb(img, out_path)

    print(f"Done. Output saved to: {out_path}")
    print(f"Original: {W}x{H}  ->  New: {img.shape[1]}x{img.shape[0]}")
    print(f"Total removed seam cost: {total_cost:.2f}")
    print(f"Runtime: {dt:.2f} s")


if __name__ == "__main__":
    main()
