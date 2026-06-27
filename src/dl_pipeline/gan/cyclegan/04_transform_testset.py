#!/usr/bin/env python3
"""
STEP 2.6 - Translate the labeled S2 test set through CycleGAN G_B.

Goal: build a YOLO dataset in which full S2 test frames are "repainted" into the
S1 style (generator G_B: S2->S1), while the LABELS stay the same (CycleGAN does
not move the larva). We then compute the mAP of the S1-only YOLO model on this
dataset and compare it against the baseline (raw S2 = domain shift,
larva mAP@0.5 = 9.1%).

The generator is loaded DIRECTLY from a .pth checkpoint (resnet_9blocks), without
test.py - this lets us translate full 1920x1080 frames (test.py is built for 256
crops).

Input:
    --gen      checkpoints/<run>/latest_net_G_B.pth   (S2->S1)
    --src      data/session2_only/test                (images/ + labels/)
Output:
    --out      data/session2_test_cyclegan/           (images/ + labels/ + data.yaml)

Usage (on a GPU):
    export CUDA_VISIBLE_DEVICES=0
    python src/dl_pipeline/gan/cyclegan/04_transform_testset.py \
        --gen   <cyclegan_repo>/checkpoints/<run>/latest_net_G_B.pth \
        --src   data/session2_only/test \
        --out   data/session2_test_cyclegan \
        --load-size 1024
"""

import argparse
import functools
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# ResNet generator definition (matches pytorch-CycleGAN-and-pix2pix)
# ============================================================================

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias):
        super().__init__()
        conv = [nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
                norm_layer(dim), nn.ReLU(True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
                norm_layer(dim)]
        self.conv_block = nn.Sequential(*conv)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """ResNet generator with 9 blocks - identical to --netG resnet_9blocks."""

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super().__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        use_bias = True
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, bias=use_bias),
                 norm_layer(ngf), nn.ReLU(True)]
        n_down = 2
        for i in range(n_down):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]
        mult = 2 ** n_down
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer, use_bias)]
        for i in range(n_down):
            mult = 2 ** (n_down - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def load_generator(gen_path: str, device: str) -> ResnetGenerator:
    net = ResnetGenerator()
    state = torch.load(gen_path, map_location=device)
    if hasattr(state, "_metadata"):
        del state._metadata
    net.load_state_dict(state)
    net.to(device).eval()
    return net


def transform_image(net, img_bgr: np.ndarray, load_size: int, device: str) -> np.ndarray:
    """Translate a BGR image (any size) -> target style; return BGR at original size."""
    h0, w0 = img_bgr.shape[:2]
    # Input size must be a multiple of 4 (2 downsampling layers)
    side = (load_size // 4) * 4
    img = cv2.resize(img_bgr, (side, side), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ten = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    ten = (ten - 0.5) / 0.5  # normalize to [-1, 1]
    ten = ten.to(device)
    with torch.no_grad():
        out = net(ten)
    out = (out.squeeze(0).cpu().permute(1, 2, 0).numpy() + 1) / 2.0  # to [0, 1]
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return cv2.resize(out_bgr, (w0, h0), interpolation=cv2.INTER_CUBIC)


def main():
    ap = argparse.ArgumentParser(description="STEP 2.6 - translate the S2 test set through CycleGAN G_B")
    ap.add_argument("--gen", required=True, help="Generator G_B checkpoint (.pth, S2->S1)")
    ap.add_argument("--src", required=True, help="S2 test-set folder (contains images/ and labels/)")
    ap.add_argument("--out", required=True, help="Output dataset folder")
    ap.add_argument("--load-size", type=int, default=1024, help="Translation size (default 1024)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  (if cpu -> run on a GPU node!)")

    src = Path(args.src)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    net = load_generator(args.gen, device)
    print(f"Generator loaded: {args.gen}")

    imgs = sorted((src / "images").glob("*.jpg")) + sorted((src / "images").glob("*.png"))
    print(f"Frames to translate: {len(imgs)}")

    for i, img_path in enumerate(imgs):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  skipping (not loaded): {img_path.name}")
            continue
        fake = transform_image(net, img, args.load_size, device)
        cv2.imwrite(str(out / "images" / img_path.name), fake)

        # Label unchanged (the larva does not move after translation)
        lbl = (src / "labels" / img_path.with_suffix(".txt").name)
        if lbl.exists():
            shutil.copy2(lbl, out / "labels" / lbl.name)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(imgs)}")

    # data.yaml pointing to this dataset (for mAP evaluation)
    yaml_text = (
        f"names:\n- dish\n- larva\nnc: 2\n"
        f"test: {out / 'images'}\n"
        f"train: {out / 'images'}\n"
        f"val: {out / 'images'}\n"
    )
    (out / "data.yaml").write_text(yaml_text)

    print(f"\nDONE. Dataset: {out}")
    print(f"  images: {len(list((out/'images').glob('*')))}, "
          f"labels: {len(list((out/'labels').glob('*.txt')))}")
    print(f"  data.yaml: {out/'data.yaml'}")


if __name__ == "__main__":
    main()
