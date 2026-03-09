import pandas as pd
import numpy as np

'''can modify .csv path here'''
path = "data/b l-194136.csv"

with open(path, encoding="ISO-8859-1") as f:
    lines = f.readlines()

data = []

for line in lines:
    parts = line.strip().split(",")

    if len(parts) < 2:
        continue

    try:
        lam = float(parts[0])
        I = float(parts[1])
    except:
        continue

    if 380 <= lam <= 780:
        data.append((lam, I))

df = pd.DataFrame(data, columns=["lambda", "I"])


x_bar = np.zeros(781)
y_bar = np.zeros(781)
z_bar = np.zeros(781)

path = "CIE_xyz_1931_2deg.csv"
with open(path) as f:
    for line in f:
        parts = line.strip().split(",")

        if len(parts) != 4:
            continue

        lam = int(parts[0])

        # 只取 380–780 nm
        if 380 <= lam <= 780:

            x_bar[lam] = float(parts[1])
            y_bar[lam] = float(parts[2])
            z_bar[lam] = float(parts[3])


lam_array = df["lambda"].to_numpy()
I_array = df["I"].to_numpy()
X = 0
Y = 0
Z = 0
for l, i_val in zip(lam_array, I_array):
    l = int(l)

    X += i_val * x_bar[l]
    Y += i_val * y_bar[l]
    Z += i_val * z_bar[l]

x = X/(X+Y+Z)
y = Y/(X+Y+Z)

print("x = ",x)
print("y = ",y)
