"""
Usage:
    python eval_image_quality.py --gen path/to/generated.png --gt path/to/clean_gt.png
    
    # Batch mode: pass directories
    python eval_image_quality.py --gen path/to/gen_dir --gt path/to/gt_dir
"""

import argparse
import os
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def load_image_np(path):
    """Load image as numpy array in [0, 1] range."""
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float64) / 255.0


def load_image_tensor(path):
    """Load image as torch tensor in [-1, 1] range for LPIPS."""
    img = Image.open(path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    # HWC -> CHW, [0,1] -> [-1,1]
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return tensor


def evaluate_pair(gen_path, gt_path, lpips_fn):
    """Compute PSNR, SSIM, LPIPS for a single image pair."""
    gen_np = load_image_np(gen_path)
    gt_np = load_image_np(gt_path)

    # Resize if dimensions don't match
    if gen_np.shape != gt_np.shape:
        gt_img = Image.open(gt_path).convert("RGB")
        gen_img = Image.open(gen_path).convert("RGB")
        gen_img = gen_img.resize(gt_img.size, Image.LANCZOS)
        gen_np = np.array(gen_img).astype(np.float64) / 255.0
        gt_np = np.array(gt_img).astype(np.float64) / 255.0

    # PSNR
    psnr = compute_psnr(gt_np, gen_np, data_range=1.0)

    # SSIM
    ssim = compute_ssim(gt_np, gen_np, data_range=1.0, channel_axis=2)

    # LPIPS
    gen_tensor = load_image_tensor(gen_path)
    gt_tensor = load_image_tensor(gt_path)
    if gen_tensor.shape != gt_tensor.shape:
        gen_tensor = torch.nn.functional.interpolate(
            gen_tensor, size=gt_tensor.shape[2:], mode="bilinear", align_corners=False
        )
    with torch.no_grad():
        lpips_val = lpips_fn(gen_tensor, gt_tensor).item()

    return psnr, ssim, lpips_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", required=True, help="Generated image or directory")
    parser.add_argument("--gt", required=True, help="Clean GT image or directory")
    parser.add_argument("--lpips_net", default="vgg", choices=["vgg", "alex"])
    args = parser.parse_args()

    lpips_fn = lpips.LPIPS(net=args.lpips_net, verbose=False)
    lpips_fn.eval()

    gen_path = Path(args.gen)
    gt_path = Path(args.gt)

    # Single image mode
    if gen_path.is_file() and gt_path.is_file():
        psnr, ssim, lpips_val = evaluate_pair(gen_path, gt_path, lpips_fn)
        print(f"PSNR:  {psnr:.4f}")
        print(f"SSIM:  {ssim:.4f}")
        print(f"LPIPS: {lpips_val:.4f}")
        return

    # Batch mode
    if gen_path.is_dir() and gt_path.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        gen_files = sorted([f for f in gen_path.iterdir() if f.suffix.lower() in exts])

        all_psnr, all_ssim, all_lpips = [], [], []
        for gf in gen_files:
            gt_file = gt_path / gf.name
            if not gt_file.exists():
                print(f"[SKIP] No GT match for {gf.name}")
                continue

            psnr, ssim, lpips_val = evaluate_pair(gf, gt_file, lpips_fn)
            all_psnr.append(psnr)
            all_ssim.append(ssim)
            all_lpips.append(lpips_val)
            print(f"{gf.name:40s}  PSNR={psnr:.2f}  SSIM={ssim:.4f}  LPIPS={lpips_val:.4f}")

        if all_psnr:
            print("\n" + "=" * 60)
            print(f"{'Average':40s}  PSNR={np.mean(all_psnr):.2f}  SSIM={np.mean(all_ssim):.4f}  LPIPS={np.mean(all_lpips):.4f}")
            print(f"{'Std':40s}  PSNR={np.std(all_psnr):.2f}  SSIM={np.std(all_ssim):.4f}  LPIPS={np.std(all_lpips):.4f}")
            print(f"Total: {len(all_psnr)} image pairs evaluated")
        return

    print("Error: --gen and --gt must both be files or both be directories.")


if __name__ == "__main__":
    main()