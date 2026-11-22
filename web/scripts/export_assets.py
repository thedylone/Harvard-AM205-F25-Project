"""Export the sample images and the repo's saved Kodak results for the page.

Run from web/:  python3 scripts/export_assets.py
Writes public/samples/*.png and src/data/kodak-results.json.
"""
import json
import os
import pickle

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..")
ROOT = os.path.join(WEB, "..")


def gray(rgb):
    # rgb_to_grayscale from code/main.py
    return np.floor(np.sum(rgb[..., :3] * np.array([0.299, 0.587, 0.114]), axis=2)).astype(np.uint8)


def save(name, arr):
    Image.fromarray(arr).save(os.path.join(WEB, "public", "samples", name), optimize=True)
    print(name, arr.shape)


os.makedirs(os.path.join(WEB, "public", "samples"), exist_ok=True)

import scipy.datasets  # noqa: E402
from skimage import data  # noqa: E402

# scipy.datasets.face(gray=True) is the test image used in code/main.py (768×1024);
# halve it with a 2×2 box filter so it fits the 512px working size.
face = scipy.datasets.face(gray=True).astype(np.float64)
face = face.reshape(384, 2, 512, 2).mean(axis=(1, 3))
save("raccoon.png", np.round(face).astype(np.uint8))
save("cameraman.png", data.camera())
save("astronaut.png", gray(data.astronaut().astype(np.float64)))

# ---- Kodak (Med-Res) results saved by code/sample_run.py ----
# Each method maps to a flat list [cr_img0, metric_img0, cr_img1, metric_img1, ...].
bins = np.linspace(0, 4, 33)
out = {"bins": bins.tolist(), "methods": {}}
for metric in ["psnr", "ssim"]:
    with open(os.path.join(ROOT, "results", f"med_res_{metric}.pkl"), "rb") as f:
        res = pickle.load(f)
    for method, flat in res.items():
        crs = np.concatenate([np.asarray(flat[i], float) for i in range(0, len(flat), 2)])
        vals = np.concatenate([np.asarray(flat[i], float) for i in range(1, len(flat), 2)])
        n_images = len(flat) // 2
        lc = np.log10(crs)
        idx = np.digitize(lc, bins) - 1
        pts = []
        for b in range(len(bins) - 1):
            v = vals[(idx == b) & np.isfinite(vals)]
            if len(v) < 3:
                continue
            mean = float(v.mean())
            ci = float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
            pts.append({"cr": float(10 ** lc[idx == b].mean()), "mean": mean, "lo": mean - ci, "hi": mean + ci, "n": int(len(v))})
        out["methods"].setdefault(method, {})[metric] = pts
        out["images"] = n_images
with open(os.path.join(WEB, "src", "data", "kodak-results.json"), "w") as f:
    json.dump(out, f, indent=1)
print({m: {k: len(v) for k, v in d.items()} for m, d in out["methods"].items()}, out["images"])

# ---- Hero strip: the raccoon in four vertical bands (original, DWT, DFT, PCA at CR = 50) ----
import sys  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "code"))
import pywt  # noqa: E402
from compression import methods  # noqa: E402,F401  (ensures the repo module imports)
from numpy.fft import fft2, ifft2  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

img = face  # 384×512, float
ratio = 50.0
c_arr, c_sl = pywt.coeffs_to_array(pywt.wavedec2(img, "bior4.4"))
kf = int(c_arr.size / ratio)
t = np.partition(np.abs(c_arr.flatten()), -kf)[-kf]
dwt = pywt.waverec2(pywt.array_to_coeffs(pywt.threshold(c_arr, t, mode="hard"), c_sl, output_format="wavedec2"), "bior4.4")
F = fft2(img)
kf = int(F.size / ratio)
t = np.partition(np.abs(F).flatten(), -kf)[-kf]
dft = np.real(ifft2(np.where(np.abs(F) >= t, F, 0)))
h, w = img.shape
k = max(1, int(min(h, w, h * w / (h + w)) / ratio))
pca = PCA(n_components=k)
pc = pca.inverse_transform(pca.fit_transform(img))
bands = [img, dwt, dft, pc]
out = np.zeros_like(img)
edges = np.linspace(0, w, 5).astype(int)
for i, b in enumerate(bands):
    out[:, edges[i]:edges[i + 1]] = b[:, edges[i]:edges[i + 1]]
Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8)).save(
    os.path.join(WEB, "public", "hero-strip.webp"), "WEBP", quality=90, method=6
)
print("hero strip", out.shape, "pca k", k)
