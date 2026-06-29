import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\koyamaharuki\OneDrive\デスクトップ\vscode\研究用")
AUDIO_DIR = BASE_DIR / '楽曲データwav'

# 正規化後の保存先
output_dir = BASE_DIR / "music_norm_wav"
output_dir.mkdir(exist_ok=True)

# 実験用なら -18.0 あたりが無難
target_lufs = -18.0

# 最大ピークを -1 dBFS 以下に抑える
peak_limit_db = -1.0
peak_limit_amp = 10 ** (peak_limit_db / 20)

# 特徴量抽出コードに合わせて、BASE_DIR以下のwavを再帰的に取得
audio_files = sorted(AUDIO_DIR.rglob("*.wav"))

# 出力フォルダ内のファイルを再処理しないように除外
audio_files = [
    p for p in audio_files
    if output_dir not in p.parents
]

if len(audio_files) == 0:
    print(f"wavファイルが見つかりません: {BASE_DIR}")

log_rows = []

for file_path in audio_files:
    file_name = file_path.name

    data, rate = sf.read(file_path)

    # 無音・読み込み異常対策
    if data.size == 0 or np.max(np.abs(data)) == 0:
        print(f"スキップ: {file_name} は無音または読み込み異常です")
        continue

    # ラウドネス計測
    meter = pyln.Meter(rate)
    loudness_before = meter.integrated_loudness(data)

    # -inf や nan 対策
    if not np.isfinite(loudness_before):
        print(f"スキップ: {file_name} はLUFSを計測できませんでした")
        continue

    # LUFS正規化
    norm_data = pyln.normalize.loudness(
        data,
        loudness_before,
        target_lufs
    )

    # LUFS正規化直後のピーク確認
    peak_after_lufs = np.max(np.abs(norm_data))
    peak_limited = False

    # クリップ防止：-1 dBFSを超える場合はピークを下げる
    if peak_after_lufs > peak_limit_amp:
        norm_data = pyln.normalize.peak(norm_data, peak_limit_db)
        peak_limited = True

    # 最終確認
    loudness_after = meter.integrated_loudness(norm_data)
    peak_after = np.max(np.abs(norm_data))

    # wavとして保存する
    # mp3に再エンコードするより、研究用にはwav保存の方が安全
    output_path = output_dir / f"{file_path.stem}.wav"
    sf.write(output_path, norm_data, rate)

    log_rows.append({
        "filename_original": file_path.name,
        "filename_normalized": output_path.name,
        "original_path": str(file_path),
        "normalized_path": str(output_path),
        "loudness_before_lufs": loudness_before,
        "loudness_after_lufs": loudness_after,
        "peak_after_lufs_normalize": peak_after_lufs,
        "peak_after_final": peak_after,
        "peak_limited": peak_limited,
    })

    print(
        f"{file_name}: "
        f"{loudness_before:.2f} LUFS -> {loudness_after:.2f} LUFS, "
        f"peak={peak_after:.3f}, "
        f"peak_limited={peak_limited}, "
        f"saved={output_path.name}"
    )

# ログ保存
log_df = pd.DataFrame(log_rows)
log_path = output_dir / "loudness_normalization_log.csv"

log_df.to_csv(
    log_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nラウドネス正規化が完了しました。")
print(f"ログを保存しました: {log_path}")