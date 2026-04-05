import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ===== 參數設定 =====
stripe_period = 4   # 60 LPI 對應 MacBook ~224 PPI → 約 4 pixels
left_pixels = 2     # 左眼佔 2 pixels
right_pixels = 2    # 右眼佔 2 pixels

# ===== 螢幕解析度（MacBook Air M3 13"）=====
SCREEN_W = 2560
SCREEN_H = 1664

# ===== 讀取圖片 =====
img1 = Image.open("image_left.png").convert("RGB")
img2 = Image.open("image_right.png").convert("RGB")

# ===== resize 成螢幕大小（關鍵🔥）=====
img1 = img1.resize((SCREEN_W, SCREEN_H), Image.BICUBIC)
img2 = img2.resize((SCREEN_W, SCREEN_H), Image.BICUBIC)

img1 = np.array(img1)
img2 = np.array(img2)

H, W, _ = img1.shape

# ===== 建立輸出圖 =====
output = np.zeros_like(img1)

# ===== Interlacing（符合 60 LPI）=====
for col in range(W):
    if (col % stripe_period) < left_pixels:
        output[:, col, :] = img1[:, col, :]   # Left view
    else:
        output[:, col, :] = img2[:, col, :]   # Right view

# ===== 儲存（一定要用這個看🔥）=====
Image.fromarray(output).save("interlaced_fullscreen.png")

# ===== Fullscreen 顯示 =====
fig = plt.figure(figsize=(16,10))
ax = plt.axes([0,0,1,1])   # 填滿整個畫面
ax.imshow(output)
ax.axis('off')

# 👉 強制全螢幕（不同系統可能略有差異）
try:
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
except:
    pass

plt.show()