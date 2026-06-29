import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from scipy.signal import find_peaks


def safe_mean(x):
    if x.size == 0:
        return np.nan
    return np.mean(x)


def safe_std(x):
    if x.size == 0:
        return np.nan
    return np.std(x)


def count_band_peaks_from_energy(S_energy, freq, freq_min, freq_max, mask):
    """
    S_energy: shape = (周波数ビン数, フレーム数)
    freq: 各周波数ビンのHz
    mask: 集計対象の時間フレーム
    """
    freq_range = (freq >= freq_min) & (freq < freq_max)

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


def extract_whole_song_features(file_path, SR=22050, total_sec=32):
    """
    1曲全体の特徴量を計算する関数
    """
    y, sr = librosa.load(file_path, sr=SR, mono=True, duration=total_sec)

    n_fft = 2048
    hop_length = 512
    n_mfcc = 20

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
    spectral_centroids = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]

    # MFCC
    mel = librosa.feature.melspectrogram(S=S_power, sr=sr, n_mels=128)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)

    # ゼロクロス率
    zcr = librosa.feature.zero_crossing_rate(
        y=y,
        frame_length=n_fft,
        hop_length=hop_length
    )[0]

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

    # 全体特徴量を計算
    features = {}
    features['filename'] = str(file_path)

    # 強度指標
    features['rms_mean'] = safe_mean(rms)
    features['rms_std'] = safe_std(rms)
    energy_threshold = safe_mean(rms)
    if len(rms) > 0:
        features['low_energy'] = np.sum(rms < energy_threshold) / len(rms)
    else:
        features['low_energy'] = np.nan

    # 周波数指標
    features['spectral_centroids_mean'] = safe_mean(spectral_centroids)
    features['spectral_centroids_std'] = safe_std(spectral_centroids)
    features['spectral_bandwidth_mean'] = safe_mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = safe_std(spectral_bandwidth)
    features['spectral_rolloff_mean'] = safe_mean(spectral_rolloff)
    features['spectral_rolloff_std'] = safe_std(spectral_rolloff)

    # MFCC
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    for j in range(n_mfcc):
        features[f'mfcc_mean_{j}'] = mfcc_mean[j]
        features[f'mfcc_std_{j}'] = mfcc_std[j]

    # ゼロクロス率
    features['zcr_mean'] = safe_mean(zcr)
    features['zcr_std'] = safe_std(zcr)

    # リズム指標
    n_frames = S.shape[1]
    frame_times = librosa.frames_to_time(
        np.arange(n_frames),
        sr=sr,
        hop_length=hop_length
    )
    mask_all = np.ones(n_frames, dtype=bool)

    features['low_peaks'] = count_band_peaks_from_energy(
        S_perc_energy, freq, 0, 250, mask_all
    )
    features['middle_peaks'] = count_band_peaks_from_energy(
        S_perc_energy, freq, 250, 800, mask_all
    )
    features['high_peaks'] = count_band_peaks_from_energy(
        S_perc_energy, freq, 800, 2000, mask_all
    )

    features['low_peak_rate'] = features['low_peaks'] / total_sec
    features['middle_peak_rate'] = features['middle_peaks'] / total_sec
    features['high_peak_rate'] = features['high_peaks'] / total_sec

    # 和声指標
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    for j, note in enumerate(note_names):
        features[f'chroma_mean_{note}'] = chroma_mean[j]
        features[f'chroma_std_{note}'] = chroma_std[j]

    return features


def extract_acoustic_features_by_segments(file_path, SR=22050, segment_sec=4, total_sec=32):
    """
    1曲全体で特徴量を計算し、その後segment_sec秒ごとに集計する関数
    """

    # ========== 音声読み込み ==========
    y, sr = librosa.load(file_path, sr=SR, mono=True, duration=total_sec)

    n_fft = 2048
    hop_length = 512
    n_mfcc = 20

    # ========== 1曲全体で特徴量を計算 ==========

    # STFTを1回計算して使い回す
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S = np.abs(D)
    S_power = S ** 2

    # フレーム時刻
    n_frames = S.shape[1]
    frame_times = librosa.frames_to_time(
        np.arange(n_frames),
        sr=sr,
        hop_length=hop_length
    )

    # ========== 強度指標 ==========
    rms = librosa.feature.rms(
        y=y,
        frame_length=n_fft,
        hop_length=hop_length
    )[0]

    # ========== 周波数指標 ==========
    spectral_centroids = librosa.feature.spectral_centroid(
        S=S,
        sr=sr
    )[0]

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        S=S,
        sr=sr
    )[0]

    spectral_rolloff = librosa.feature.spectral_rolloff(
        S=S,
        sr=sr
    )[0]

    # MFCC
    # 先にメルスペクトログラムを1曲全体で作る
    mel = librosa.feature.melspectrogram(
        S=S_power,
        sr=sr,
        n_mels=128
    )

    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(mel),
        n_mfcc=n_mfcc
    )

    # ゼロクロス率
    zcr = librosa.feature.zero_crossing_rate(
        y=y,
        frame_length=n_fft,
        hop_length=hop_length
    )[0]

    # ========== 和声・リズム分離 ==========
    y_harmonic, y_percussive = librosa.effects.hpss(
        y,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # HPSSは音声をharmonic成分とpercussive成分に分解する処理です。
    # Librosa公式でも、入力音声をharmonic/percussiveに分解し、出力波形長を入力と揃える関数として説明されています。

    # パーカッシブ成分のSTFT
    D_perc = librosa.stft(
        y_percussive,
        n_fft=n_fft,
        hop_length=hop_length
    )
    S_perc_energy = np.abs(D_perc) ** 2

    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # ハーモニック成分からクロマ
    chroma = librosa.feature.chroma_stft(
        y=y_harmonic,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # ========== segment_sec秒ごとに集計 ==========
    features_list = []

    num_segments = int(total_sec / segment_sec)

    for seg_idx in range(num_segments):
        start_time = seg_idx * segment_sec
        end_time = (seg_idx + 1) * segment_sec

        mask = (frame_times >= start_time) & (frame_times < end_time)

        features = {}

        features['filename'] = str(file_path)
        features['segment_index'] = seg_idx
        features['start_time'] = start_time
        features['end_time'] = end_time

        # ========== 強度指標 ==========
        rms_seg = rms[mask]
        features['rms_mean'] = safe_mean(rms_seg)
        features['rms_std'] = safe_std(rms_seg)

        energy_threshold = safe_mean(rms_seg)
        if len(rms_seg) > 0:
            features['low_energy'] = np.sum(rms_seg < energy_threshold) / len(rms_seg)
        else:
            features['low_energy'] = np.nan

        # ========== 周波数指標 ==========
        spectral_centroids_seg = spectral_centroids[mask]
        features['spectral_centroids_mean'] = safe_mean(spectral_centroids_seg)
        features['spectral_centroids_std'] = safe_std(spectral_centroids_seg)

        spectral_bandwidth_seg = spectral_bandwidth[mask]
        features['spectral_bandwidth_mean'] = safe_mean(spectral_bandwidth_seg)
        features['spectral_bandwidth_std'] = safe_std(spectral_bandwidth_seg)

        spectral_rolloff_seg = spectral_rolloff[mask]
        features['spectral_rolloff_mean'] = safe_mean(spectral_rolloff_seg)
        features['spectral_rolloff_std'] = safe_std(spectral_rolloff_seg)

        # MFCC
        mfcc_seg = mfcc[:, mask]
        mfcc_mean = np.mean(mfcc_seg, axis=1)
        mfcc_std = np.std(mfcc_seg, axis=1)

        for j in range(n_mfcc):
            features[f'mfcc_mean_{j}'] = mfcc_mean[j]
            features[f'mfcc_std_{j}'] = mfcc_std[j]

        # ゼロクロス率
        zcr_seg = zcr[mask]
        features['zcr_mean'] = safe_mean(zcr_seg)
        features['zcr_std'] = safe_std(zcr_seg)

        # ========== リズム指標 ==========
        features['low_peaks'] = count_band_peaks_from_energy(
            S_perc_energy,
            freq,
            0,
            250,
            mask
        )

        features['middle_peaks'] = count_band_peaks_from_energy(
            S_perc_energy,
            freq,
            250,
            800,
            mask
        )

        features['high_peaks'] = count_band_peaks_from_energy(
            S_perc_energy,
            freq,
            800,
            2000,
            mask
        )

        features['low_peak_rate'] = features['low_peaks'] / segment_sec
        features['middle_peak_rate'] = features['middle_peaks'] / segment_sec
        features['high_peak_rate'] = features['high_peaks'] / segment_sec

        # ========== 和声指標 ==========
        chroma_seg = chroma[:, mask]
        chroma_mean = np.mean(chroma_seg, axis=1)
        chroma_std = np.std(chroma_seg, axis=1)

        for j, note in enumerate(note_names):
            features[f'chroma_mean_{note}'] = chroma_mean[j]
            features[f'chroma_std_{note}'] = chroma_std[j]

        features_list.append(features)

    return features_list


# ========== データ読み込み ==========

SR = 44100
total_sec = 32

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = Path(BASE_DIR / 'music_norm')

# 1. 全体特徴量を計算して保存
print("=" * 50)
print("全体特徴量を計算中...")
print("=" * 50)

whole_song_features_list = []

for i, file_path in enumerate(BASE_DIR.glob('**/*.mp3')):
    print(f'{i + 1}つ目処理中（全体特徴量）：{file_path}')
    features = extract_whole_song_features(
        file_path=file_path,
        SR=SR,
        total_sec=total_sec
    )
    whole_song_features_list.append(features)

df_whole = pd.DataFrame(whole_song_features_list)

print(df_whole.head())
print(f"全体特徴量: {df_whole.shape}")

# 全体特徴量をCSVに保存
output_path_whole = BASE_DIR / "acoustic_features_whole.csv"

df_whole.to_csv(output_path_whole, index=False, encoding="utf-8-sig")
print(f"全体特徴量をCSVに保存しました: {output_path_whole}")

# 2. 複数のセグメント幅で処理
#segment_secs = [1.0]

#for segment_sec in segment_secs:
    #print("\n" + "=" * 50)
    #print(f"{segment_sec}秒ごとのセグメント特徴量を計算中...")
    #print("=" * 50)

    #features_list = []

    #for i, file_path in enumerate(AUDIO_DIR.glob('**/*.mp3')):
        #print(f'{i + 1}つ目処理中（{segment_sec}秒セグメント）：{file_path}')
        #features_list.extend(
            #extract_acoustic_features_by_segments(
                #file_path=file_path,
                #SR=SR,
                #segment_sec=segment_sec,
                #total_sec=total_sec
            #)
        #)

    #df_segment = pd.DataFrame(features_list)

    #print(df_segment.head())
    #print(f"セグメント特徴量（{segment_sec}秒）: {df_segment.shape}")

    # セグメント特徴量をCSVに保存
    #segment_str = str(segment_sec).replace('.', '_')
    #output_path_segment = BASE_DIR / "acoustic_features_segment.csv"

    #df_segment.to_csv(output_path_segment, index=False, encoding="utf-8-sig")

    #df_segment.to_csv(output_path_segment, index=False, encoding="utf-8-sig")
    #print(f"セグメント特徴量（{segment_sec}秒）をCSVに保存しました: {output_path_segment}")

print("\n" + "=" * 50)
print("すべての処理が完了しました！")
print("=" * 50)