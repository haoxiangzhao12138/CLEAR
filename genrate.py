"""
Image Quality Evaluation

Mode 1 - With GT (reference-based + no-reference):
    python eval_image_quality.py --gen generated.png --gt clean.png

Mode 2 - Without GT (no-reference only, measures clarity alone):
    python eval_image_quality.py --gen generated.png

Batch mode:
    python eval_image_quality.py --gen ./sft_states/ --gt ./clean_gt/
    python eval_image_quality.py --gen ./rl_states/

Dependencies:
    pip install lpips scikit-image pillow pyiqa torch tqdm
"""

import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import torch


def load_image_np(path):
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float64) / 255.0


def load_image_tensor_lpips(path):
    """Load as [-1, 1] tensor for LPIPS."""
    img = Image.open(path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return tensor


def compute_reference_metrics(gen_path, gt_path, lpips_fn):
    """PSNR, SSIM, LPIPS — all require GT."""
    from skimage.metrics import peak_signal_noise_ratio as calc_psnr
    from skimage.metrics import structural_similarity as calc_ssim

    gen_np = load_image_np(gen_path)
    gt_np = load_image_np(gt_path)

    # resize if mismatch
    if gen_np.shape != gt_np.shape:
        gen_img = Image.open(gen_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        gen_img = gen_img.resize(gt_img.size, Image.LANCZOS)
        gen_np = np.array(gen_img).astype(np.float64) / 255.0
        gt_np = np.array(gt_img).astype(np.float64) / 255.0

    psnr = calc_psnr(gt_np, gen_np, data_range=1.0)
    ssim = calc_ssim(gt_np, gen_np, data_range=1.0, channel_axis=2)

    gen_t = load_image_tensor_lpips(gen_path)
    gt_t = load_image_tensor_lpips(gt_path)
    if gen_t.shape != gt_t.shape:
        gen_t = torch.nn.functional.interpolate(
            gen_t, size=gt_t.shape[2:], mode="bilinear", align_corners=False
        )
    with torch.no_grad():
        lpips_val = lpips_fn(gen_t, gt_t).item()

    return {"PSNR": psnr, "SSIM": ssim, "LPIPS": lpips_val}


def compute_noreference_metrics(img_path, metrics_cache):
    """
    No-reference metrics via pyiqa. No GT needed.

    BRISQUE: lower = better quality  (typical range 0~100, good < 30)
    NIQE:    lower = better quality  (typical range 2~10, good < 5)
    MUSIQ:   higher = better quality (typical range 0~100)
    CLIPIQA: higher = better quality (typical range 0~1)
    """
    import pyiqa

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for name in ["brisque", "niqe", "musiq", "clipiqa"]:
        if name not in metrics_cache:
            metrics_cache[name] = pyiqa.create_metric(name, device=device)
        score = metrics_cache[name](str(img_path)).item()
        results[name.upper()] = score

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", default="/root/CLEAR/VLMEvalKit/vis_merged/reason_interleave_grpo_20260330_190814_new_KL_per_token", help="Generated image or directory")
    parser.add_argument("--gt", default=None, help="GT image or directory (optional)")
    args = parser.parse_args()

    has_gt = args.gt is not None
    gen_path = Path(args.gen)
    gt_path = Path(args.gt) if has_gt else None

    # init LPIPS
    lpips_fn = None
    if has_gt:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).eval()

    # init pyiqa cache
    nr_cache = {}

    # collect files
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if gen_path.is_file():
        gen_files = [gen_path]
    else:
        gen_files = sorted(f for f in gen_path.iterdir() if f.suffix.lower() in exts)

    all_results = []
    skipped = 0

    from tqdm import tqdm
    for gf in tqdm(gen_files, desc="Evaluating", ncols=80):
        result = {}

        # reference-based
        if has_gt:
            gt_file = gt_path / gf.name if gt_path.is_dir() else gt_path
            if not gt_file.exists():
                skipped += 1
                continue
            ref = compute_reference_metrics(gf, gt_file, lpips_fn)
            result.update(ref)

        # no-reference
        try:
            noref = compute_noreference_metrics(gf, nr_cache)
            result.update(noref)
        except ImportError:
            print("\n[WARN] pyiqa not installed. Run: pip install pyiqa")
            break

        all_results.append(result)

    # summary
    print("\n" + "=" * 50)
    if all_results:
        keys = [k for k in all_results[0]]
        for k in keys:
            v = [r[k] for r in all_results if k in r]
            print(f"{k:10s}  mean={np.mean(v):.4f}  std={np.std(v):.4f}")
        print(f"Total: {len(all_results)} images evaluated")
        if skipped:
            print(f"Skipped: {skipped} (no matching GT)")
    else:
        print("No images evaluated.")


if __name__ == "__main__":
    main()