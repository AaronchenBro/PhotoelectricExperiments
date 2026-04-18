import pandas as pd
import matplotlib.pyplot as plt
import os

data_dir = "data"
save_dir = "picture"

os.makedirs(save_dir, exist_ok=True)

for file in os.listdir(data_dir):

    # 只處理 output（排除 linear / sat）
    if ("linear" in file) or ("sat" in file):
        continue

    filepath = os.path.join(data_dir, file)
    df = pd.read_csv(filepath)

    VGS = df["栅极电压(V)"]
    VDS = df["漏极电压(V)"]
    ID  = df["漏极电流(A)"]

    plt.figure(figsize=(6,4))

    VGS_rounded = VGS.round(1)
    unique_vgs = sorted(VGS_rounded.unique())

    for vgs in unique_vgs:
        mask = (VGS_rounded == vgs)
        plt.plot(VDS[mask], ID[mask], label=f"VGS={vgs:.1f}V")

    plt.xlabel("VDS (V)")
    plt.ylabel("ID (A)")
    plt.title(f"Output Characteristics ({file})")
    plt.legend()
    plt.grid()

    # 檔名處理（去掉 .csv）
    name = file.replace(".csv", "_output.png")
    save_path = os.path.join(save_dir, name)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()   # ⚠️ 很重要（避免記憶體爆）

print("All plots saved to /picture/")