from pathlib import Path
import pandas as pd
import numpy as np
import os

# المسار الحالي للسكربت
BASE_DIR = Path(__file__).resolve().parent

# نرجع مجلدين لفوق ثم ندخل data/raw/daphnet
folder_path = BASE_DIR.parent.parent / "data" / "raw" / "daphnet"

all_data = []

for file_path in folder_path.glob("*.txt"):
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df = df[[0, 1, 3, 2, 10]]
    df.columns = ["time", "Acc_X", "Acc_Y", "Acc_Z", "label"]
    df["file"] = file_path.name
    all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)

# مسارات الحفظ
save_folder = BASE_DIR.parent.parent / "data" / "filtered"
csv_path = save_folder / "Daphnet_ankle_acc.csv"
npz_path = save_folder / "Daphnet_ankle_acc.npz"

final_df.to_csv(csv_path, index=False, encoding="utf-8")
np.savez(npz_path, data=final_df.to_numpy())

print("✅ تم إنشاء ملفات CSV و NPZ بنجاح!")
