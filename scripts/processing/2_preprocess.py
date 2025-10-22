import numpy as np
from scipy.signal import butter, filtfilt
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
def segment_signal(X, y, fs=64, window_sec=2):
    w = fs * window_sec   # 64 * 3 = 192
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
    input_dir  = r"..\data\filtered"
    output_dir = r"..\data\processed"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".npz"):
            file_path = os.path.join(input_dir, file)
            print("Processing:", file_path)

            data = np.load(file_path, allow_pickle=True)["data"]

            # Extract columns (time, x, y, z, label, filename)
            acc_x = data[:, 1].astype(float)
            acc_y = data[:, 2].astype(float)
            acc_z = data[:, 3].astype(float)
            labels = data[:, 4].astype(int)

            # Remove label 0 samples
            mask = labels != 0
            acc_x, acc_y, acc_z, labels = acc_x[mask], acc_y[mask], acc_z[mask], labels[mask]

            # Apply filtering
            acc_x = butter_lowpass_filter(acc_x)
            acc_y = butter_lowpass_filter(acc_y)
            acc_z = butter_lowpass_filter(acc_z)

            # Stack into (N, 3)
            X = np.column_stack((acc_x, acc_y, acc_z))

            # Segment
            X_windows, y_windows = segment_signal(X, labels)

            # Save
            save_path = os.path.join(output_dir, file.replace(".npz", "p_daphnet_detection_2sec.npz"))
            np.savez(save_path, X=X_windows, y=y_windows)

            print("Saved:", save_path)
            print("X shape:", X_windows.shape, "| y shape:", y_windows.shape)

    print("Done.")
