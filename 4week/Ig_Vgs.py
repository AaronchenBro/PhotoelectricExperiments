import pandas as pd
import matplotlib.pyplot as plt
import os

data_dir = "data"
save_dir = "picture"
os.makedirs(save_dir, exist_ok=True)


def extract_IG(file):
    df = pd.read_csv(os.path.join(data_dir, file))

    VGS = df["栅极电压(V)"].round(1)
    IG  = df["栅极电流(A)"]

    # 每個 VGS 取平均
    grouped = df.groupby(VGS)["栅极电流(A)"].mean()

    return grouped.sort_index()


# ===== 找檔案 =====
linear_files = [f for f in os.listdir(data_dir) if "no_light_linear" in f]
sat_files    = [f for f in os.listdir(data_dir) if "no_light_sat" in f]


# =========================
# 🔹 Linear IG–VGS
# =========================
plt.figure(figsize=(6,4))

for file in linear_files:
    curve = extract_IG(file)
    plt.semilogy(curve.index, curve.values, marker='o', label=file)

plt.xlabel("VGS (V)")
plt.ylabel("IG (A)")
plt.title("IG–VGS (Linear Region) - No Light")
plt.legend()
plt.grid()

plt.savefig(os.path.join(save_dir, "no_light_IG_VGS_linear.png"), dpi=300)
plt.show()


# =========================
# 🔹 Saturation IG–VGS
# =========================
plt.figure(figsize=(6,4))

for file in sat_files:
    curve = extract_IG(file)
    plt.semilogy(curve.index, curve.values, marker='o', label=file)

plt.xlabel("VGS (V)")
plt.ylabel("IG (A)")
plt.title("IG–VGS (Saturation Region) - No Light")
plt.legend()
plt.grid()

plt.savefig(os.path.join(save_dir, "no_light_IG_VGS_sat.png"), dpi=300)
plt.show()