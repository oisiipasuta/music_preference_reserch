import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from scipy.signal import find_peaks


def count_band_peaks_from_energy_window_mask(
    S_energy,
    freq,
    freq_min,
    freq_max,
    mask,
    frame_times=None,
    hop_sec=None,
    min_peak_distance_sec=0.08,
    prominence_std=0.3,
    return_rate=False,
    duration_sec=None
):
    """
    すでに作成済みのmaskを使って、帯域別ピーク数をカウントする関数。

    Parameters
    ----------
    S_energy : np.ndarray
        shape = (周波数ビン数, フレーム数)
        パーカッシブ成分などのエネルギースペクトログラム。

    freq : np.ndarray
        各周波数ビンのHz。

    freq_min, freq_max : float
        集計する周波数帯域。

    mask : np.ndarray
        対象時間窓に対応するフレームだけ True の配列。
        例: mask = (frame_times >= start_sec) & (frame_times < end_sec)

    frame_times : np.ndarray, optional
        各フレームの秒数。hop_sec計算やduration計算に使う。

    hop_sec : float, optional
        1フレームの秒数。例: hop_length / sr。
        frame_timesがある場合は不要。

    min_peak_distance_sec : float
        近すぎるピークを重複カウントしないための最小間隔。

    prominence_std : float
        ピークとして認める突出度。

    return_rate : bool
        Trueならピーク数ではなく、1秒あたりピーク数を返す。

    duration_sec : float, optional
        return_rate=True のときに使う区間長。
        4秒窓なら duration_sec=4.0 を渡してもよい。

    Returns
    -------
    int or float
        ピーク数、またはピーク密度。
    """

    S_energy = np.asarray(S_energy)
    freq = np.asarray(freq)
    mask = np.asarray(mask, dtype=bool)

    # 周波数方向の長さを安全にそろえる
    n_freq = min(S_energy.shape[0], len(freq))
    S_energy = S_energy[:n_freq, :]
    freq = freq[:n_freq]

    # 周波数帯域を選択
    freq_range = (freq >= freq_min) & (freq < freq_max)

    if not np.any(freq_range):
        return 0

    # 時間フレーム方向の長さを安全にそろえる
    n_frames = min(S_energy.shape[1], len(mask))

    if frame_times is not None:
        frame_times = np.asarray(frame_times)
        n_frames = min(n_frames, len(frame_times))
        frame_times = frame_times[:n_frames]

    S_energy = S_energy[:, :n_frames]
    mask = mask[:n_frames]

    if not np.any(mask):
        return 0

    # 指定帯域の平均エネルギー時系列
    band_energy = np.mean(S_energy[freq_range, :], axis=0)

    # maskで指定時間窓だけ取り出す
    band_energy_segment = band_energy[mask]

    if len(band_energy_segment) < 3:
        return 0

    if np.std(band_energy_segment) == 0:
        return 0

    # フレーム間隔を秒で決める
    if hop_sec is None:
        if frame_times is not None and len(frame_times) >= 2:
            hop_sec = np.median(np.diff(frame_times))
        else:
            hop_sec = 0.01

    # 最小ピーク間隔をフレーム数に変換
    distance_frames = max(1, int(round(min_peak_distance_sec / hop_sec)))

    # ピーク検出
    peaks, _ = find_peaks(
        band_energy_segment,
        height=np.mean(band_energy_segment),
        prominence=np.std(band_energy_segment) * prominence_std,
        distance=distance_frames
    )

    peak_count = len(peaks)

    if return_rate:
        if duration_sec is None:
            duration_sec = np.sum(mask) * hop_sec

        if duration_sec <= 0:
            return 0

        return peak_count / duration_sec

    return peak_count


def safe_mean(x):
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))


def extract_sliding_window_features(
    file_path,
    SR=44100,
    total_sec=32,
    window_sec=4,
    step_sec=0.1
):
    """
    1曲を4秒窓でスライドしながら特徴量抽出する関数。
    例：
    0-4秒
    1-5秒
    2-6秒
    ...
    28-32秒
    """

    y, sr = librosa.load(file_path, sr=SR, duration=total_sec)

    n_fft = 2048
    hop_length = 512
    n_mfcc = 20

    # STFT
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
    spectral_centroids = librosa.feature.spectral_centroid(
        S=S,
        sr=sr,
        n_fft=n_fft
    )[0]

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        S=S,
        sr=sr,
        n_fft=n_fft
    )[0]

    spectral_rolloff = librosa.feature.spectral_rolloff(
        S=S,
        sr=sr,
        n_fft=n_fft
    )[0]

    # MFCC
    mel = librosa.feature.melspectrogram(
        S=S_power,
        sr=sr,
        n_mels=128
    )

    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(mel),
        n_mfcc=n_mfcc
    )

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(
        y=y,
        frame_length=n_fft,
        hop_length=hop_length
    )[0]

    # HPSS
    y_harmonic, y_percussive = librosa.effects.hpss(
        y,
        n_fft=n_fft,
        hop_length=hop_length
    )

    D_perc = librosa.stft(
        y_percussive,
        n_fft=n_fft,
        hop_length=hop_length
    )

    S_perc_energy = np.abs(D_perc) ** 2

    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=y_harmonic,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )

    note_names = ['C', 'C_sharp', 'D', 'D_sharp', 'E', 'F', 'F_sharp', 'G', 'G_sharp', 'A', 'A_sharp', 'B']

    # 各フレームの時刻を作成
    n_frames = S.shape[1]
    frame_times = librosa.frames_to_time(
        np.arange(n_frames),
        sr=sr,
        hop_length=hop_length
    )

    # 0, 1, 2, ..., 28秒のように窓開始時刻を作る
    max_start = total_sec - window_sec
    n_steps = int(round(max_start / step_sec))

    window_starts = np.arange(n_steps + 1) * step_sec
    window_starts = np.round(window_starts, 3)

    filename = str(file_path.stem)
    rows = []

    print(f"スライディング窓特徴量を抽出中: {filename}")

    for start_sec in window_starts:
        end_sec = start_sec + window_sec

        # ここが最重要
        # start_sec〜end_secに対応するフレームだけTrueになる
        mask = (frame_times >= start_sec) & (frame_times < end_sec)

        if not np.any(mask):
            continue

        features = {}
        features["filename"] = filename
        features["window_start_sec"] = float(start_sec)
        features["window_end_sec"] = float(end_sec)
        features["window_sec"] = float(window_sec)

        # 強度・周波数指標
        features["rms"] = safe_mean(rms[mask])
        features["spectral_centroids"] = safe_mean(spectral_centroids[mask])
        features["spectral_bandwidth"] = safe_mean(spectral_bandwidth[mask])
        features["spectral_rolloff"] = safe_mean(spectral_rolloff[mask])
        features["zcr"] = safe_mean(zcr[mask])

        # MFCC
        for j in range(n_mfcc):
            features[f"mfcc_{j}"] = safe_mean(mfcc[j, mask])

        # リズム指標
        features["low_peaks"] = count_band_peaks_from_energy_window_mask(
            S_energy=S_perc_energy,
            freq=freq,
            freq_min=0,
            freq_max=500,
            mask=mask,
            frame_times=frame_times,
            min_peak_distance_sec=0.08,
            prominence_std=0.3,
            return_rate=True
        )
        features["middle_peaks"] = count_band_peaks_from_energy_window_mask(
            S_energy=S_perc_energy,
            freq=freq,
            freq_min=500,
            freq_max=2000,
            mask=mask,
            frame_times=frame_times,
            min_peak_distance_sec=0.08,
            prominence_std=0.3,
            return_rate=True
        )
        features["high_peaks"] = count_band_peaks_from_energy_window_mask(
            S_energy=S_perc_energy,
            freq=freq,
            freq_min=2000,
            freq_max=8000,
            mask=mask,
            frame_times=frame_times,
            min_peak_distance_sec=0.08,
            prominence_std=0.3,
            return_rate=True
        )
        features["super_high_peaks"] = count_band_peaks_from_energy_window_mask(
            S_energy=S_perc_energy,
            freq=freq,
            freq_min=8000,
            freq_max=20000,
            mask=mask,
            frame_times=frame_times,
            min_peak_distance_sec=0.08,
            prominence_std=0.3,
            return_rate=True
        )

        # 和声指標
        for j, note in enumerate(note_names):
            features[f"chroma_mean_{note}"] = safe_mean(chroma[j, mask])

        rows.append(features)

    return rows

SR = 44100
total_sec = 32
window_sec = 4
step_sec = 0.1

BASE_DIR = Path(r"C:\Users\koyamaharuki\OneDrive\デスクトップ\vscode\研究用")  # 例: Path("C:/Users/username/Documents")
AUDIO_DIR = BASE_DIR / "music_norm_wav"

sliding_features_list = []

for i, file_path in enumerate(AUDIO_DIR.glob("*.wav")):
    print(f"{i + 1}つ目処理中：{file_path}")

    rows = extract_sliding_window_features(
        file_path=file_path,
        SR=SR,
        total_sec=total_sec,
        window_sec=window_sec,
        step_sec=step_sec
    )

    sliding_features_list.extend(rows)

df_sliding = pd.DataFrame(sliding_features_list)

print(df_sliding.head())
print(df_sliding.shape)

output_path = BASE_DIR / "acoustic_features_sliding_4sec_1sec.csv"
df_sliding.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"保存しました: {output_path}")