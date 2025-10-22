import numpy as np
from scipy.signal import butter, filtfilt
from pathlib import Path
import pandas as pd
import os

# ==========================
# Butterworth Low-Pass Filter
# ==========================
def butter_lowpass_filter(data, cutoff=10, fs=64, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# ==========================
# Segmentation (3s, No Overlap)
# ==========================
def segment_signal(X, y, fs=64, window_sec=3):
    w = fs * window_sec
    X_windows = []
    y_windows = []

    for start in range(0, len(X), w):
        end = start + w
        if end <= len(X):
            window = X[start:end]
            labels = y[start:end]

            # Skip windows containing label 0
            if 0 in labels:
                continue

            # Assign window label
            if 2 in labels:
                y_windows.append(2)
            else:
                y_windows.append(1)

            X_windows.append(window)

    return np.array(X_windows), np.array(y_windows)

# ==========================
# Main
# ==========================
if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    input_dir  = BASE_DIR / "data" / "filtered"
    output_dir = BASE_DIR / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("INPUT DIR:", input_dir)
    print("OUTPUT DIR:", output_dir)

    for file in os.listdir(input_dir):
        if file.endswith(".npz"):
            file_path = input_dir / file
            print("Processing:", file_path)

            data = np.load(file_path, allow_pickle=True)["data"]

            acc_x = data[:, 1].astype(float)
            acc_y = data[:, 2].astype(float)
            acc_z = data[:, 3].astype(float)
            labels = data[:, 4].astype(int)

            # Remove label 0
            mask = labels != 0
            acc_x, acc_y, acc_z, labels = acc_x[mask], acc_y[mask], acc_z[mask], labels[mask]

            # Apply Butterworth filter
            acc_x = butter_lowpass_filter(acc_x)
            acc_y = butter_lowpass_filter(acc_y)
            acc_z = butter_lowpass_filter(acc_z)

            # Stack into (N, 3)
            X = np.column_stack((acc_x, acc_y, acc_z))

            # Segment
            X_windows, y_windows = segment_signal(X, labels)

            # ==== Save NPZ ====
            save_npz = output_dir / "p_daphnet_detection_3sec.npz"
            np.savez(save_npz, X=X_windows, y=y_windows)

            # ==== Save CSV ====
            X_flat = X_windows.reshape(X_windows.shape[0], -1)
            df = pd.DataFrame(X_flat)
            df["label"] = y_windows
            save_csv = output_dir / "p_daphnet_detection_3sec.csv"
            df.to_csv(save_csv, index=False)

            print("Saved:", save_npz)
            print("Saved:", save_csv)
            print("X shape:", X_windows.shape, "| y shape:", y_windows.shape)

    print("\nDone. All files saved successfully!")
