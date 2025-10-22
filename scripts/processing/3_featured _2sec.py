# scripts/processing/3_featured_2sec.py
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
from pathlib import Path
import pandas as pd
import os

# =============================
# Magnitude
# =============================
def compute_magnitude(window_xyz):
    return np.sqrt(np.sum(window_xyz**2, axis=1))

# =============================
# Spectral Helpers
# =============================
def spectral_entropy(Pxx):
    Pxx_norm = Pxx / np.sum(Pxx)
    return -np.sum(Pxx_norm * np.log2(Pxx_norm + 1e-12))

def spectral_rolloff(freqs, Pxx, roll_percent=0.85):
    cumulative_energy = np.cumsum(Pxx)
    threshold = roll_percent * cumulative_energy[-1]
    idx = np.where(cumulative_energy >= threshold)[0][0]
    return freqs[idx]

# =============================
# 27 features per window
# =============================
def extract_features(sig, fs=64):
    N = len(sig)

    # ---- Time Domain ----
    mean_val = np.mean(sig)
    std_val = np.std(sig)
    var_val = np.var(sig)
    rms_val = np.sqrt(np.mean(sig**2))
    min_val = np.min(sig)
    max_val = np.max(sig)
    range_val = max_val - min_val
    skew_val = skew(sig)
    kurt_val = kurtosis(sig)
    zcr = ((sig[:-1] * sig[1:]) < 0).sum() / max(1, N)
    energy = np.sum(sig**2)
    sma = np.sum(np.abs(sig)) / N

    # ---- Frequency Domain ----
    freqs = fftfreq(N, 1/fs)
    fft_vals = np.abs(fft(sig))
    Pxx = fft_vals[:N//2] ** 2
    freqs = freqs[:N//2]

    total_power = np.sum(Pxx) if np.sum(Pxx) != 0 else 1e-12
    spec_centroid = np.sum(freqs * Pxx) / total_power
    spec_entropy = spectral_entropy(Pxx)
    spec_energy = total_power / N
    dom_freq = freqs[np.argmax(Pxx)] if len(Pxx) > 0 else 0.0
    mean_freq = np.mean(freqs) if len(freqs) > 0 else 0.0
    median_freq = np.median(freqs) if len(freqs) > 0 else 0.0
    freq_var = np.var(Pxx)
    freq_kurt = kurtosis(Pxx)
    freq_skew = skew(Pxx)
    dc_comp = mean_val
    peak_freq = dom_freq
    roll_off = spectral_rolloff(freqs, Pxx) if len(Pxx) > 0 else 0.0
    flux = np.mean(np.diff(Pxx)**2) if len(Pxx) > 1 else 0.0
    flatness = np.exp(np.mean(np.log(Pxx + 1e-12))) / (np.mean(Pxx) + 1e-12)
    bandwidth = np.sqrt(np.sum(((freqs - spec_centroid)**2) * Pxx) / total_power)

    return [
        mean_val, std_val, var_val, rms_val, min_val, max_val, range_val,
        skew_val, kurt_val, zcr, energy, sma,
        spec_centroid, spec_entropy, spec_energy, dom_freq, mean_freq, median_freq,
        freq_var, freq_kurt, freq_skew, dc_comp, peak_freq, roll_off, flux, flatness, bandwidth
    ]

# =============================
# Main
# =============================
if __name__ == "__main__":

    # اعتمد نفس نظام المسارات الديناميكية كما في السكربتات السابقة
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    input_dir = BASE_DIR / "data" / "featured"
    # اسم ملف الإدخال المتوقع (نافذة 2s)
    file_name = "p_daphnet_detection_2sec.npz"

    # مسارات الحفظ الموحدة للـ features (NPZ + CSV)
    save_npz = input_dir / "f_daphnet_detection_2sec.npz"
    save_csv = input_dir / "f_daphnet_detection_2sec.csv"

    # فحص وجود ملف الإدخال قبل المتابعة
    input_path = input_dir / file_name
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("Loading:", input_path)
    data = np.load(input_path)
    X = data["X"]   # shape expected: (N, 128, 3) for 2s @64Hz
    y = data["y"]

    all_feats = []
    for i in range(len(X)):
        sig_mag = compute_magnitude(X[i])
        feats = extract_features(sig_mag)
        all_feats.append(feats)

    X_feats = np.array(all_feats)
    y_feats = np.array(y)

    # Save NPZ
    np.savez(save_npz, X=X_feats, y=y_feats)

    # Save CSV
    df = pd.DataFrame(X_feats)
    df["label"] = y_feats
    df.to_csv(save_csv, index=False)

    print("Saved:", save_npz)
    print("Saved:", save_csv)
    print("Final Shapes -> X:", X_feats.shape, "| y:", y_feats.shape)
