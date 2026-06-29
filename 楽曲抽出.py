import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# 1. データ読み込み
# =========================

BASE_PATH = Path(r"C:\Users\koyamaharuki\OneDrive\デスクトップ\vscode\研究用\acoustic_features_segment_with_metadata_melody_split_no_std_peakrate.csv")

df = pd.read_csv(BASE_PATH)

# 不要列を削除
df = df.drop(df.columns[49:56], axis=1)

# =========================
# 2. PCAに入れないメタデータ列を指定
# =========================

exclude_cols = [
    'file_stem',
    'segment_index',

    'harmony_level',
    'harmony_progression',

    'melody_type',
    'melody_strength',

    'rhythm_level',
    'rhythm_pattern',

]

# 実際に存在する列だけに絞る
exclude_cols = [col for col in exclude_cols if col in df.columns]

# =========================
# 3. PCAに使う音響特徴量列を作成
# =========================

feature_cols = [
    col for col in df.select_dtypes(include='number').columns
    if col not in exclude_cols
]

print("PCAに使う特徴量数:", len(feature_cols))
print("PCAに使う特徴量:")
print(feature_cols)

print("\n除外された数値メタデータ候補:")
excluded_numeric_cols = [
    col for col in df.select_dtypes(include='number').columns
    if col in exclude_cols
]
print(excluded_numeric_cols)

# =========================
# 4. メタデータ列を作成
# =========================

meta_cols = [
    col for col in df.columns
    if col not in feature_cols and col != 'file_stem'
]

print("\nメタデータ列:")
print(meta_cols)

# =========================
# 5. 曲ごとに平均化
# =========================

df_mean_group = (
    df.groupby('file_stem')
    .agg({
        **{col: 'mean' for col in feature_cols},
        **{col: 'first' for col in meta_cols}
    })
    .reset_index()
)

# =========================
# 6. メタデータの表記を変換
# =========================

if 'harmony_level' in df_mean_group.columns:
    harmony_mapping = {
        '小': 'h_simple',
        '中': 'h_complex',
        '大': 'h_very_complex'
    }
    df_mean_group['harmony_level'] = df_mean_group['harmony_level'].map(harmony_mapping)

if 'melody_type' in df_mean_group.columns:
    melody_mapping = {
        '上昇': 'm_up',
        '下降': 'm_down',
        '山型': 'm_peak',
        '一定': 'm_constant'
    }
    df_mean_group['melody_type'] = df_mean_group['melody_type'].map(melody_mapping)

if 'rhythm_level' in df_mean_group.columns:
    rhythm_mapping = {
        '小': 'r_simple',
        '中': 'r_complex',
        '大': 'r_very_complex'
    }
    df_mean_group['rhythm_level'] = df_mean_group['rhythm_level'].map(rhythm_mapping)

# =========================
# 7. 結合メタデータを作成
# =========================

if {'harmony_level', 'harmony_progression'}.issubset(df_mean_group.columns):
    df_mean_group['harmony_all'] = (
        df_mean_group['harmony_level'].fillna('unknown') + '_' +
        df_mean_group['harmony_progression'].astype(str).fillna('unknown')
    )

if {'melody_type', 'melody_strength'}.issubset(df_mean_group.columns):
    df_mean_group['melody_all'] = (
        df_mean_group['melody_type'].fillna('unknown') + '_' +
        df_mean_group['melody_strength'].astype(str).fillna('unknown')
    )

if {'rhythm_level', 'rhythm_pattern'}.issubset(df_mean_group.columns):
    df_mean_group['rhythm_all'] = (
        df_mean_group['rhythm_level'].fillna('unknown') + '_' +
        df_mean_group['rhythm_pattern'].astype(str).fillna('unknown')
    )

display(df_mean_group.head())
df_mean_group.info()

# =========================
# 8. PCA
# =========================

# 念のため欠損値を平均値で補完
X = df_mean_group[feature_cols].copy()
X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n = 6
pca = PCA(n_components=n)

X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(
    X_pca,
    columns=[f'PC{i+1}' for i in range(n)]
)

df_pca['file_stem'] = df_mean_group['file_stem'].values

# =========================
# 9. 寄与率
# =========================

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

print("Explained Variance Ratio:")
print(explained_variance_ratio)

print("\nCumulative Explained Variance:")
print(cumulative_explained_variance)

# =========================
# 10. Loading
# =========================

loading = pca.components_.T * np.sqrt(pca.explained_variance_)

loading_df = pd.DataFrame(
    loading,
    index=feature_cols,
    columns=[f'PC{i+1}' for i in range(n)]
)

display(loading_df)

# =========================
# 11. PCAにメタデータが混入していないか確認
# =========================

suspect_keywords = [
    'melody',
    'harmony',
    'rhythm',
    'segment',
    'condition',
    'file',
    'metadata'
]

suspect_cols = [
    col for col in feature_cols
    if any(keyword in col.lower() for keyword in suspect_keywords)
]

print("\nメタデータっぽい列がPCAに混入していないか確認:")
if len(suspect_cols) == 0:
    print("問題なし：メタデータっぽい列は見つかりませんでした。")
else:
    print("注意：以下の列がPCAに入っています。必要なら exclude_cols に追加してください。")
    print(suspect_cols)

# =========================
# 12. Loadingのヒートマップ
# =========================

plt.figure(figsize=(20, max(8, len(loading_df) * 0.4)))

sns.heatmap(
    loading_df,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    yticklabels=True,
    xticklabels=True
)

plt.title('PCA Loadings Heatmap')
plt.xlabel('Principal Components')
plt.ylabel('Original Features')

plt.yticks(rotation=0, fontsize=9)
plt.xticks(rotation=45, fontsize=10)

plt.tight_layout()
plt.show()

# =========================
# 13. PCAスコアの散布図を層別化,harmony_all, melody_all, rhythm_all
# =========================

for meta_col in ['harmony_all', 'melody_all', 'rhythm_all']:
    if meta_col in df_mean_group.columns:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=df_pca.merge(df_mean_group[[meta_col, 'file_stem']], on='file_stem'),
            x='PC1',
            y='PC2',
            hue=meta_col,
            s=100,
            alpha=0.7
        )
        plt.title(f'PCA Score Plot Colored by {meta_col}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title=meta_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=df_pca.merge(df_mean_group[[meta_col, 'file_stem']], on='file_stem'),
            x='PC1',
            y='PC3',
            hue=meta_col,
            s=100,
            alpha=0.7
        )
        plt.title(f'PCA Score Plot Colored by {meta_col}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 3')
        plt.legend(title=meta_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


# =========================
# 14. クラスタリング
# =========================

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# クラスタリングに使うPCAスコア
# 寄与率の大きいPCを重視するなら X_pca のままでOK
X_cluster = X_pca

cluster_range = range(2, 10)
silhouette_scores = []

for n_clusters in cluster_range:
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(X_cluster)
    silhouette_avg = silhouette_score(X_cluster, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Number of clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(list(cluster_range), silhouette_scores, marker='o')
plt.title('Silhouette Score for KMeans Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(list(cluster_range))
plt.grid()
plt.tight_layout()
plt.show()

# =========================
# 15. クラスタ数を指定
# =========================

# シルエット最大にする場合
# optimal_clusters = list(cluster_range)[np.argmax(silhouette_scores)]

# 研究上、6クラスタで解釈・刺激選出したい場合はこちら
optimal_clusters = 6

kmeans = KMeans(
    n_clusters=optimal_clusters,
    random_state=42,
    n_init=10
)

df_pca['Cluster'] = kmeans.fit_predict(X_cluster)

# =========================
# 16. PCAスコアとメタデータを結合
# =========================

metadata_cols = ['harmony_all', 'melody_all', 'rhythm_all']

df_pca_meta = df_pca.merge(
    df_mean_group[['file_stem'] + metadata_cols],
    on='file_stem',
    how='left'
)

# =========================
# 17. クラスタリング結果の可視化
# =========================

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_pca_meta,
    x='PC1',
    y='PC2',
    hue='Cluster',
    palette='Set2',
    s=100,
    alpha=0.7
)
plt.title(f'PCA Score Plot Colored by KMeans Clusters (k={optimal_clusters})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_pca_meta,
    x='PC1',
    y='PC3',
    hue='Cluster',
    palette='Set2',
    s=100,
    alpha=0.7
)
plt.title(f'PCA Score Plot Colored by KMeans Clusters (k={optimal_clusters})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 3')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# =========================
# 18. クラスタごとのPC平均
# =========================

cluster_means = df_pca_meta.groupby('Cluster')[
    [f'PC{i+1}' for i in range(n)]
].mean()

print("\nCluster Means:")
display(cluster_means)

# ヒートマップで見ると解釈しやすい
plt.figure(figsize=(10, 5))
sns.heatmap(
    cluster_means,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0
)
plt.title('Mean PCA Scores by Cluster')
plt.xlabel('Principal Components')
plt.ylabel('Cluster')
plt.tight_layout()
plt.show()

# =========================
# 19. クラスタごとのメタデータ分布
# =========================

def analyze_metadata_distribution(df, metadata_cols):
    for meta_col in metadata_cols:
        if meta_col in df.columns:
            plt.figure(figsize=(12, 6))
            sns.countplot(
                data=df,
                x='Cluster',
                hue=meta_col
            )
            plt.title(f'Distribution of {meta_col} Across Clusters')
            plt.xlabel('Cluster')
            plt.ylabel('Count')
            plt.legend(title=meta_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

analyze_metadata_distribution(df_pca_meta, metadata_cols)


#クラスターデータをcsvに保存したい
output_csv_path = Path(r"C:\Users\koyamaharuki\OneDrive\デスクトップ\vscode\研究用\pca_cluster_results.csv")
df_pca_meta.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
print(f"クラスタリング結果を保存しました: {output_csv_path}")