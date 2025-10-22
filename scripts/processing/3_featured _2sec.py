import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
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
# 27 features per 2s window
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
    zcr = ((sig[:-1] * sig[1:]) < 0).sum() / N
    energy = np.sum(sig**2)
    sma = np.sum(np.abs(sig)) / N

    # ---- Frequency Domain ----
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

# =============================
# Main
# =============================
if __name__ == "__main__":
    input_dir = r"E:\boon\graduation project\fog-detection\data\processed"
    file_name = "p_daphnet_detection_2sec.npz"
    save_path = os.path.join(input_dir, "f_daphnet_detection_2sec.npz")

    data = np.load(os.path.join(input_dir, file_name))
    X = data["X"]  # shape: (N, 128, 3)
    y = data["y"]

    all_feats = []
    for i in range(len(X)):
        sig_mag = compute_magnitude(X[i])
        feats = extract_features(sig_mag)
        all_feats.append(feats)

    X_feats = np.array(all_feats)
    y_feats = np.array(y)

    np.savez(save_path, X=X_feats, y=y_feats)

    print("Saved:", save_path)
    print("Final Shapes -> X:", X_feats.shape, "| y:", y_feats.shape)
