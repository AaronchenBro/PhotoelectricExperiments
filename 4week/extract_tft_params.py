import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

data_dir = "data"


# =========================
# 🔧 讀取 transfer curve
# =========================
def load_transfer(file):
    df = pd.read_csv(os.path.join(data_dir, file))

    VGS = df["栅极电压(V)"].round(1)
    ID  = df["漏极电流(A)"].abs()

    # 每個 VGS 取最大 ID（代表 transfer）
    grouped = df.groupby(VGS)["漏极电流(A)"].max()

    vgs = grouped.index.values.astype(float)
    id_ = grouped.values.astype(float)

    return vgs, id_


# =========================
# 🔧 找最佳線性區（用 R²）
# =========================
def find_linear_region(vgs, y):
    best_r2 = -1
    best_slice = slice(0, 4)

    for i in range(len(vgs) - 4):
        x = vgs[i:i+4]
        y_seg = y[i:i+4]

        a, b = np.polyfit(x, y_seg, 1)
        y_fit = a * x + b

        ss_res = np.sum((y_seg - y_fit) ** 2)
        ss_tot = np.sum((y_seg - np.mean(y_seg)) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

        if r2 > best_r2:
            best_r2 = r2
            best_slice = slice(i, i+4)

    return best_slice


# =========================
# 🔹 Vth (Linear)
# =========================
def calc_vth_linear(vgs, id_):
    gm = np.gradient(id_, vgs)
    idx = np.argmax(gm)

    start = max(0, idx - 2)
    end = min(len(vgs), idx + 2)

    a, b = np.polyfit(vgs[start:end], id_[start:end], 1)
    vth = -b / a

    return vth


# =========================
# 🔹 Vth (Saturation)
# =========================
def calc_vth_sat(vgs, id_):
    sqrt_id = np.sqrt(id_)

    gm = np.gradient(sqrt_id, vgs)
    idx = np.argmax(gm)

    start = max(0, idx - 2)
    end = min(len(vgs), idx + 2)

    a, b = np.polyfit(vgs[start:end], sqrt_id[start:end], 1)
    vth = -b / a

    return vth


# =========================
# 🔹 SS（用最線性區）
# =========================
def calc_ss(vgs, id_):
    log_id = np.log10(id_)

    region = find_linear_region(vgs, log_id)

    a, _ = np.polyfit(vgs[region], log_id[region], 1)

    ss_mv = (1 / a) * 1000
    return ss_mv


# =========================
# 🔹 On/Off
# =========================
def calc_onoff(id_):
    ion = np.max(id_)
    ioff = np.min(id_)
    return ion / ioff


# =========================
# 🔍 找檔案
# =========================
linear_files = [f for f in os.listdir(data_dir) if "no_light_linear" in f]
sat_files    = [f for f in os.listdir(data_dir) if "no_light_sat" in f]

print("Linear files:", linear_files)
print("Sat files   :", sat_files)


# =========================
# 📊 計算
# =========================
results = []

# Linear
for f in linear_files:
    vgs, id_ = load_transfer(f)

    vth = calc_vth_linear(vgs, id_)
    ss = calc_ss(vgs, id_)
    onoff = calc_onoff(id_)

    results.append({
        "file": f,
        "type": "linear",
        "Vth (V)": vth,
        "SS (mV/dec)": ss,
        "On/Off": onoff
    })


# Saturation
for f in sat_files:
    vgs, id_ = load_transfer(f)

    vth = calc_vth_sat(vgs, id_)
    ss = calc_ss(vgs, id_)
    onoff = calc_onoff(id_)

    results.append({
        "file": f,
        "type": "sat",
        "Vth (V)": vth,
        "SS (mV/dec)": ss,
        "On/Off": onoff
    })


# =========================
# 📋 輸出
# =========================
results_df = pd.DataFrame(results)

print("\n===== TFT Parameters (No Light) =====")
print(results_df)

results_df.to_csv("no_light_parameters.csv", index=False)
print("\nSaved to no_light_parameters.csv")

plt.semilogy(vgs, id_)