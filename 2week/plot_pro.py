import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# x位置
x = [-120,-110,-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,
     0,10,20,30,40,50,60,70,80,90]

R = [80,130,200,280,350,420,400,300,200,140,100,90,
     80,70,60,60,65,70,75,80,85,90]

G = [50,50,50,45,45,50,60,80,120,150,180,200,
     220,300,350,400,360,280,230,210,160,120]

RG = [r + g for r, g in zip(R, G)]

# ===== 畫圖 =====
plt.figure(figsize=(8,5))

plt.plot(x, R, marker='o', label='Red')
plt.plot(x, G, marker='^', label='Green')
plt.plot(x, RG, marker='s', label='Red + Green')

plt.xlabel("Position")
plt.ylabel("Irradiance")
plt.title("R, G, and Summed Distribution")

plt.legend()
plt.grid(True)


# 原始資料
x = np.array([-120,-110,-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,
              0,10,20,30,40,50,60,70,80,90])

R = np.array([80,130,200,280,350,420,400,300,200,140,100,90,
              80,70,60,60,65,70,75,80,85,90])

G = np.array([50,50,50,45,45,50,60,80,120,150,180,200,
              220,300,350,400,360,280,230,210,160,120])

RG = R + G

# ===== 建立平滑 x =====
x_smooth = np.linspace(x.min(), x.max(), 300)

# ===== spline 平滑 =====
R_smooth = make_interp_spline(x, R)(x_smooth)
G_smooth = make_interp_spline(x, G)(x_smooth)
RG_smooth = make_interp_spline(x, RG)(x_smooth)

# ===== 繪圖 =====
plt.figure(figsize=(8,5))

plt.plot(x_smooth, R_smooth, label='Red')
plt.plot(x_smooth, G_smooth, label='Green')
plt.plot(x_smooth, RG_smooth, linewidth=2, label='Red + Green')

# 原始點（可選，教學很有用）
plt.scatter(x, R)
plt.scatter(x, G)
plt.scatter(x, RG)

plt.xlabel("Position")
plt.ylabel("Irradiance")
plt.title("Figure 2: Smoothed R, G, and Sum")

plt.legend()
plt.grid(True)

# ===== 存圖 =====
plt.savefig("figure2.png", dpi=300)
