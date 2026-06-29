import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

#特徴量計算の関数を作成
def extract_acoustic_features(file_path, SR, t):
    """
    １つのオーディオファイルを指定された秒数ごとに区切り、それぞれの音響特徴量を全て抽出する関数
    入力：filepath, サンプリング周波数, 区切りたい時間（３２の約数）
    戻り値：特徴量ベクトル
    """
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    window = t * SR
    times = 32 // t

    #格納対象のデータを作成
    features_list = []

    #操作対象のデータを作成
    for i in range(times):
        features = {}
        audio = y[i*window : (i+1)*window]
        
        #格納するキー名を設定する
        features['filename'] = file_path
        features['start_time'] = i * t

        #ここから１サンプルずつ計算
        #=============強度指標======================

        #フレーム内エネルギー(平均と分散)
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)

        #Low Energy
        energy_threshold = np.mean(rms)
        low_energy_ratio = np.sum(rms < energy_threshold) / len(rms)
        features['low_energy'] = low_energy_ratio


        #=============周波数指標====================

        #スペクトル重心
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=SR)[0]
        features['spectral_centroids_mean'] = np.mean(spectral_centroids)
        features['spectral_centroids_std'] = np.std(spectral_centroids)

        #周波数帯域幅
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=SR)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

        #スペクトルロールオフ
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SR)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)

        #MFCC（メル周波数ケプストラム係数）
        n_mfcc = 20
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        for i in range(n_mfcc):
            features[f'mfcc_mean_{i}'] = mfcc_mean[i]
            features[f'mfcc_std_{i}'] = mfcc_std[i]
        
        #ゼロクロス率
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)


        #=============リズム指標====================

        #和声とリズムに分離
        y_harmonic, y_percussive = librosa.effects.hpss(y=audio)

        #ピーク検出のための周波数帯域分割
        n_fft = 2048
        S_energy = np.abs(librosa.stft(y_percussive, n_fft=n_fft)) ** 2
        freq = librosa.fft_frequencies(sr=SR, n_fft=n_fft)

        #低域、中域、高域のピーク数
        low_freq_range = (freq >= 0) & (freq < 250)
        middle_freq_range = (freq >= 250) & (freq < 800)
        high_freq_range = (freq >= 800) & (freq < 2000)

        from scipy.signal import find_peaks
        low_energy_mean = np.mean(S_energy[low_freq_range, :], axis=0)
        low_peaks, _ = find_peaks(low_energy_mean, height=np.mean(low_energy_mean))
        features['low_peaks'] = len(low_peaks)

        middle_energy_mean = np.mean(S_energy[middle_freq_range, :], axis=0)
        middle_peaks, _ = find_peaks(middle_energy_mean, height=np.mean(middle_energy_mean))
        features['middle_peaks'] = len(middle_peaks)

        high_energy_mean = np.mean(S_energy[high_freq_range, :], axis=0)
        high_peaks, _ = find_peaks(high_energy_mean, height=np.mean(high_energy_mean))
        features['high_peaks'] = len(high_peaks)


        #=============和声指標======================
        
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=SR)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        for i, note in enumerate(note_names):
            features[f'chroma_mean_{note}'] = chroma_mean[i]
            features[f'chroma_std_{note}'] = chroma_std[i]

        features_list.append(features)   
    return features_list


#データを取り込む
SR = 24000
t = 4
AUDIO_PATH = Path(r'C:\Users\koyamaharuki\OneDrive\デスクトップ\vscode\研究用\楽曲データdemo-20260519T090219Z-3-001')
audio_files = []

#データ型の定義
df = {
    'filename':[]
}

#データの読み込みと特徴量変換
features_list = []
for i, file_path in enumerate(AUDIO_PATH.glob('**/*.mp3')):
    print(f'{i+1}つ目処理中：{file_path}')
    features_list.extend(extract_acoustic_features(file_path, SR, t))

df = pd.DataFrame(features_list)
print(df.head())
