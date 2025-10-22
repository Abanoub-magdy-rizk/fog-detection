import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft, fftfreq
import os

# ==========================
# Magnitude
# ==========================
def compute_magnitude(window_xyz):
    return np.sqrt(np.sum(window_xyz**2, axis=1))

# ==========================
# Spectral Helpers
# ==========================
def spectral_entropy(Pxx):
    Pxx_norm = Pxx / np.sum(Pxx)
    return -np.sum(Pxx_norm * np.log2(Pxx_norm + 1e-12))

def spectral_rolloff(freqs, Pxx, roll_percent=0.85):
    cumulative_energy = np.cumsum(Pxx)
    threshold = roll_percent * cumulative_energy[-1]
    idx = np.where(cumulative_energy >= threshold)[0][0]
    return freqs[idx]

# ==========================
# 27 Features per window
# ==========================
def extract_features(sig, fs=64):
    N = len(sig)

    # ----- Time-domain -----
    mean_val = np.mean(sig)
    std_val = np.std(sig)
    var_val = np.var(sig)
    rms_val = np.sqrt(np.mean(sig**2))
    min_val = np.min(sig)
    max_val = np.max(sig)
    range_val = max_val - min_val
    skew_val = skew(sig)
    kurt_val = kurtosis(sig)
    zcr = ((sig[:-1] * sig[1:]) < 0).sum() / N
    energy = np.sum(sig**2)
    sma = np.sum(np.abs(sig)) / N

    # ----- Frequency-domain -----
    freqs = fftfreq(N, 1/fs)
    fft_vals = np.abs(fft(sig))
    Pxx = fft_vals[:N//2] ** 2
    freqs = freqs[:N//2]

    total_power = np.sum(Pxx)
    spec_centroid = np.sum(freqs * Pxx) / total_power
    spec_entropy = spectral_entropy(Pxx)
    spec_energy = total_power / N
    dom_freq = freqs[np.argmax(Pxx)]
    mean_freq = np.mean(freqs)
    median_freq = np.median(freqs)
    freq_var = np.var(Pxx)
    freq_kurt = kurtosis(Pxx)
    freq_skew = skew(Pxx)
    dc_comp = mean_val
    peak_freq = dom_freq
    roll_off = spectral_rolloff(freqs, Pxx)
    flux = np.mean(np.diff(Pxx)**2)
    flatness = np.exp(np.mean(np.log(Pxx + 1e-12))) / (np.mean(Pxx) + 1e-12)
    bandwidth = np.sqrt(np.sum(((freqs - spec_centroid)**2) * Pxx) / total_power)

    return [
        mean_val, std_val, var_val, rms_val, min_val, max_val, range_val,
        skew_val, kurt_val, zcr, energy, sma,
        spec_centroid, spec_entropy, spec_energy, dom_freq, mean_freq, median_freq,
        freq_var, freq_kurt, freq_skew, dc_comp, peak_freq, roll_off, flux, flatness, bandwidth
    ]

# ==========================
# Main Script
# ==========================
if __name__ == "__main__":
    input_dir = r"..\data\processed"
    save_path = r"..\data\featured\f_daphnet_detection_3sec.npz"

    npz_files = [f for f in os.listdir(input_dir) if f.endswith("p_daphnet_detection_3sec.npz")]
    all_feats, all_labels = [], []

    for file in npz_files:
        data = np.load(os.path.join(input_dir, file))
        X = data["X"]
        y = data["y"]

        for i in range(len(X)):
            sig_mag = compute_magnitude(X[i])
            feats = extract_features(sig_mag)
            all_feats.append(feats)
            all_labels.append(y[i])

        print("Processed:", file)

    X_feats = np.array(all_feats)
    y_feats = np.array(all_labels)

    np.savez(save_path, X=X_feats, y=y_feats)
    print("\nSaved:", save_path)
    print("Final Shapes > X:", X_feats.shape, "| y:", y_feats.shape)
