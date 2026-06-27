#!/usr/bin/env python3
"""
STEP 4.1 - Generate synthetic dish images from a trained StyleGAN2-ADA.

Loads a trained network (.pkl) and generates N 256x256 images with the truncation
trick. Output goes to --outdir as synth_NNNNNN.png + a preview grid.

REQUIRES the StyleGAN environment and the stylegan3 (or stylegan2-ada-pytorch) repo
on PYTHONPATH - the script adds the repo path itself. Run on a GPU.

Usage (GPU, StyleGAN env):
    export CUDA_VISIBLE_DEVICES=0
    python src/dl_pipeline/gan/stylegan/06_generate_synthetic.py \
        --network <stylegan_repo>/runs/<run>/network-snapshot-<best>.pkl \
        --num 1500 --trunc 0.7 \
        --outdir data/stylegan_synth/combined \
        --repo <stylegan_repo>
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: torch missing. Activate the StyleGAN environment.")
    raise SystemExit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow missing. pip install pillow")
    raise SystemExit(1)


def add_stylegan_repo_to_path(repo_hint: str = None):
    """Add the StyleGAN repo to sys.path (needed to unpickle the network)."""
    candidates = []
    if repo_hint:
        candidates.append(Path(repo_hint))
    candidates += [
        Path.cwd() / "stylegan3",
        Path.cwd() / "stylegan2-ada-pytorch",
    ]
    for c in candidates:
        if c.exists() and (c / "dnnlib").exists():
            sys.path.insert(0, str(c))
            return str(c)
    print("WARNING: could not find the StyleGAN repo automatically.")
    print("         Provide --repo /path/to/stylegan3 (or stylegan2-ada-pytorch)")
    return None


def make_grid(images, out_path: Path, n: int = 16, cell: int = 256):
    """Save a 4x4 grid of sample synthetics as a single PNG."""
    n = min(n, len(images), 16)
    if n == 0:
        return
    cols = 4
    rows = (n + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for i in range(n):
        img = images[i].resize((cell, cell))
        grid.paste(img, ((i % cols) * cell, (i // cols) * cell))
    grid.save(out_path)


def main():
    ap = argparse.ArgumentParser(description="STEP 4.1 - generate StyleGAN2-ADA synthetics")
    ap.add_argument("--network", required=True, help="Path to network-snapshot-*.pkl")
    ap.add_argument("--num", type=int, default=1500, help="Number of images to generate")
    ap.add_argument("--trunc", type=float, default=0.7, help="Truncation psi (quality vs diversity)")
    ap.add_argument("--outdir", required=True, help="Output folder for synthetics")
    ap.add_argument("--seed", type=int, default=0, help="Start seed (seeds = seed..seed+num-1)")
    ap.add_argument("--repo", type=str, default=None, help="Path to the StyleGAN repo")
    args = ap.parse_args()

    repo = add_stylegan_repo_to_path(args.repo)
    if repo:
        print(f"StyleGAN repo: {repo}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  (if cpu -> run on a GPU node!)")

    print(f"Loading network: {args.network}")
    # stylegan3/stylegan2-ada save .pkl as a dict with 'G_ema'. Prefer the repo's
    # 'legacy' loader (handles older formats); fall back to plain pickle.
    try:
        import legacy  # noqa: available after adding the repo to sys.path
        import dnnlib  # noqa
        with dnnlib.util.open_url(args.network) as f:
            G = legacy.load_network_pkl(f)["G_ema"].to(device)
    except Exception as e:
        print(f"  (legacy loader unavailable: {e} - using pickle.load)")
        with open(args.network, "rb") as f:
            G = pickle.load(f)["G_ema"].to(device)
    G.eval()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    label = torch.zeros([1, G.c_dim], device=device)  # no conditional classes
    preview = []

    print(f"Generating {args.num} images (trunc={args.trunc})...")
    for i in range(args.num):
        seed = args.seed + i
        z = torch.from_numpy(
            np.random.RandomState(seed).randn(1, G.z_dim)
        ).to(device)
        with torch.no_grad():
            img = G(z, label, truncation_psi=args.trunc, noise_mode="const")
        # [-1,1] -> [0,255] uint8, NCHW -> HWC
        img = (img.clamp(-1, 1) + 1) * (255 / 2)
        img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        pil = Image.fromarray(img, "RGB")
        pil.save(out / f"synth_{i:06d}.png")
        if i < 16:
            preview.append(pil)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{args.num}")

    make_grid(preview, out.parent / f"{out.name}_grid.png")

    print(f"\n{'='*60}")
    print(f"DONE. Synthetics: {out}  ({args.num} images)")
    print(f"  Preview grid: {out.parent / (out.name + '_grid.png')}")
    print(f"  Next step: 07_pseudo_label.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
