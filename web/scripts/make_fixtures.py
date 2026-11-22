"""Generate reference values for the TypeScript port from the repo's own code.

Run from web/:  python3 scripts/make_fixtures.py
Needs: numpy, pywavelets, scikit-learn, scikit-image, tqdm, matplotlib.
"""
import json
import os
import sys

import numpy as np
import pywt
from skimage import data, transform
from skimage.metrics import structural_similarity as ssim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
from compression import dft_compression, dwt_compression, pca_compression  # noqa: E402

out = {}
cases = {
    "tall": transform.resize(data.camera(), (70, 52), anti_aliasing=True, preserve_range=True),
    "wide": transform.resize(data.camera(), (48, 90), anti_aliasing=True, preserve_range=True),
}
ratios = np.array([1.0, 2.0, 7.5, 50.0, 300.0])
for name, img in cases.items():
    img = np.floor(img).astype(np.float64)
    m, n = img.shape
    case = {"rows": m, "cols": n, "img": img.flatten().tolist(), "ratios": ratios.tolist()}
    for method, fn in [("DWT", dwt_compression), ("DFT", dft_compression), ("PCA", pca_compression)]:
        crs, psnrs, ssims, _ = fn(img, ratios)
        case[method] = {"cr": list(map(float, crs)), "psnr": list(map(float, psnrs)), "ssim": list(map(float, ssims))}
    coeffs = pywt.wavedec2(img, "bior4.4")
    arr, _ = pywt.coeffs_to_array(coeffs)
    case["dwtShape"] = list(arr.shape)
    case["dwtApprox"] = coeffs[0].flatten().tolist()
    case["dwtDetail0"] = coeffs[1][0].flatten().tolist()
    case["recon"] = pywt.waverec2(coeffs, "bior4.4").flatten()[:20].tolist()
    noisy = img + np.random.default_rng(0).normal(0, 12, img.shape)
    case["noisy"] = noisy.flatten().tolist()
    case["ssimNoisy"] = float(ssim(img, noisy, data_range=255))
    out[name] = case

with open(os.path.join(HERE, "..", "tests", "fixtures", "reference.json"), "w") as f:
    json.dump(out, f)
print("ok", {k: (v["rows"], v["cols"], v["dwtShape"]) for k, v in out.items()})
