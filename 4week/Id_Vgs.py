import pandas as pd
import matplotlib.pyplot as plt
import os

data_dir = "data"
save_dir = "picture"
os.makedirs(save_dir, exist_ok=True)


def extract_transfer_single(file):
    df = pd.read_csv(os.path.join(data_dir, file))

    VGS = df["栅极电压(V)"].round(1)
    ID  = df["漏极电流(A)"]

    # 每個 VGS 取最大 ID
    grouped = df.groupby(VGS)["漏极电流(A)"].max()

    return grouped.sort_index()


# ===== 找檔案 =====
linear_files = [f for f in os.listdir(data_dir) if "no_light_linear" in f]
sat_files    = [f for f in os.listdir(data_dir) if "no_light_sat" in f]

# ===== Linear Plot =====
plt.figure(figsize=(6,4))

for file in linear_files:
    curve = extract_transfer_single(file)
    plt.semilogy(curve.index, curve.values, marker='o', label=file)

plt.xlabel("VGS (V)")
plt.ylabel("ID (A)")
plt.title("Transfer (Linear) - No Light")
plt.legend()
plt.grid()

plt.savefig(os.path.join(save_dir, "no_light_transfer_linear_2curves.png"), dpi=300)
plt.show()


# ===== Saturation Plot =====
plt.figure(figsize=(6,4))

for file in sat_files:
    curve = extract_transfer_single(file)
    plt.semilogy(curve.index, curve.values, marker='o', label=file)

plt.xlabel("VGS (V)")
plt.ylabel("ID (A)")
plt.title("Transfer (Saturation) - No Light")
plt.legend()
plt.grid()

plt.savefig(os.path.join(save_dir, "no_light_transfer_sat_2curves.png"), dpi=300)
plt.show()