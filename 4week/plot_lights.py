import pandas as pd
import matplotlib.pyplot as plt
import os

data_dir = "data"
save_dir = "picture"
os.makedirs(save_dir, exist_ok=True)


# =========================
# 🔧 抽取 transfer curve
# =========================
def extract_transfer(file):
    df = pd.read_csv(os.path.join(data_dir, file))

    VGS = df["栅极电压(V)"].round(1)

    # 每個 VGS 取最大 ID
    grouped = df.groupby(VGS)["漏极电流(A)"].max()

    return grouped.sort_index()


# =========================
# 🔧 畫圖函式
# =========================
def plot_compare(pattern, title, filename):
    
    files = {
        "no_light": None,
        "2.6 mW": None,
        "5 mW": None,
        "8 mW": None
    }

    # 找對應檔案
    for f in os.listdir(data_dir):
        if pattern in f:
            if "no_light" in f:
                files["no_light"] = f
            elif "2_6mw" in f:
                files["2.6 mW"] = f
            elif "5mw" in f:
                files["5 mW"] = f
            elif "8mw" in f:
                files["8 mW"] = f

    plt.figure(figsize=(6,4))

    for label, file in files.items():
        if file is None:
            continue

        curve = extract_transfer(file)

        plt.semilogy(curve.index, curve.values, marker='o', label=label)

    plt.xlabel("VGS (V)")
    plt.ylabel("ID (A)")
    plt.title(title)
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.show()


# =========================
# 🔥 畫四張圖
# =========================

plot_compare("linear0_2", "Transfer (Linear0_2)", "compare_linear0_2.png")
plot_compare("linear0_8", "Transfer (Linear0_8)", "compare_linear0_8.png")
plot_compare("sat5", "Transfer (Sat5)", "compare_sat5.png")
plot_compare("sat8", "Transfer (Sat8)", "compare_sat8.png")