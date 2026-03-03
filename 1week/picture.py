import cv2
import numpy as np
import sys

def get_center_rgb_linear(path, roi_size=200):
    # Read image (BGR)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot find {path}")

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)

    # Get center ROI
    h, w, _ = img.shape
    cy, cx = h // 2, w // 2
    half = roi_size // 2

    roi = img[cy-half:cy+half, cx-half:cx+half, :]

    # Mean RGB (assumed linear brightness)
    lr = float(np.mean(roi[:, :, 0]))
    lg = float(np.mean(roi[:, :, 1]))
    lb = float(np.mean(roi[:, :, 2]))

    return lr, lg, lb


# ===== choose picture =====
if len(sys.argv) < 2:
    print("Usage: python main.py <image_path>")
    sys.exit(1)

path = sys.argv[1]

lr, lg, lb = get_center_rgb_linear(path)

Rr, Rg, Rb = get_center_rgb_linear("Pictures/Red.jpg")
Gr, Gg, Gb = get_center_rgb_linear("Pictures/Green.jpg")
Br, Bg, Bb = get_center_rgb_linear("Pictures/Blue.jpg")

M = np.array([
    [Rr, Gr, Br],
    [Rg, Gg, Bg],
    [Rb, Gb, Bb]
], dtype=float)
M_inv = np.linalg.inv(M)

x = np.array([lr,lg,lb], dtype=float)
y = M_inv @ x
print(y)
y = np.clip(y, 1e-12, None)
y = 255 * np.exp(np.log(y) / 2.17)
y = np.round(y).astype(int)
print(path, " RGB = ", y)
