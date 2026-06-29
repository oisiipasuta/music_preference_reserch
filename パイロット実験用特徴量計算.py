import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from scipy.signal import find_peaks

def count_band_peaks_from_energy(S_energy, freq, freq_min, freq_max, mask):
    """
    S_energy: shape = (周波数ビン数, フレーム数)
    freq: 各周波数ビンのHz
    mask: 集計対象の時間フレーム
    """
    freq_range = (freq >= freq_min) & (freq < freq_max)

    if not np.any(freq_range):
        return 0

    band_energy = np.mean(S_energy[freq_range, :], axis=0)
    band_energy_segment = band_energy[mask]

    if len(band_energy_segment) == 0:
        return 0

    if np.std(band_energy_segment) == 0:
        return 0

    peaks, _ = find_peaks(
        band_energy_segment,
        height=np.mean(band_energy_segment)
    )

    return len(peaks)


def ExtractSegmentFeatures(file_path, SR=44100, total_sec=32):
    #楽曲をロードする
    y, sr = librosa.load(file_path, sr=SR, duration=total_sec)


    #初期設定値入力
    n_fft = 2048
    hop_length = 512
    n_mfcc = 20

    #フーリエ変換する
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S = np.abs(D)
    S_power = S ** 2

    # 強度指標
    rms = librosa.feature.rms(
        y=y,
        frame_length=n_fft,
        hop_length=hop_length
    )[0]

    # 周波数指標
    spectral_centroids = librosa.feature.spectral_centroid(S=S, sr=sr, n_fft=n_fft)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr, n_fft=n_fft)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, n_fft=n_fft)[0]

    # MFCC
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S_power), sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # リズム指標
    n_frames = S.shape[1]
    frame_times = librosa.frames_to_time(
        np.arange(n_frames),
        sr=sr,
        hop_length=hop_length
    )
    mask_all = np.ones(n_frames, dtype=bool)

    features['low_peaks'] = count_band_peaks_from_energy(
        S_perc_energy, freq, 0, 500, mask_all
    )
    features['middle_peaks'] = count_band_peaks_from_energy(
        S_perc_energy, freq, 500, 2000, mask_all
    )
    features['high_peaks'] = count_band_peaks_from_energy(
        S_perc_energy, freq, 2000, 8000, mask_all
    )
    features['super_high_peaks'] = count_band_peaks_from_energy(
        S_perc_energy, freq, 8000, 20000, mask_all
    )

        # 和声・リズム分離
    y_harmonic, y_percussive = librosa.effects.hpss(
        y,
        n_fft=n_fft,
        hop_length=hop_length
    )

    D_perc = librosa.stft(y_percussive, n_fft=n_fft, hop_length=hop_length)
    S_perc_energy = np.abs(D_perc) ** 2

    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    chroma = librosa.feature.chroma_stft(
        y=y_harmonic,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    #格納用のデータ構成にまとめる
    features = {}
    filename = str(file_path.stem)
    features['filename'] = filename
    print(f"特徴量を抽出中: {filename}")

SR = 44100
total_sec = 32

BASE_DIR = Path(r"C:\Users\koyamaharuki\OneDrive\デスクトップ\vscode\研究用")
AUDIO_DIR = Path(BASE_DIR / 'music_norm_wav')