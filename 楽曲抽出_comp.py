import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#データを読み込む
BASE_DIR = Path(r"C:\Users\koyamaharuki\OneDrive\デスクトップ\vscode\研究用")
df = pd.read_csv(r'C:\Users\koyamaharuki\OneDrive\デスクトップ\vscode\研究用\acoustic_features_whole.csv')

display(df.head())
print(df.shape)
print(df.info())

#dfのメタデータを分割する
def split_metadata(df):
    df['harmony'] = df['filename'].apply(lambda x: x.split('_')[0])
    df['melody'] = df['filename'].apply(lambda x: x.split('_')[1])
    df['rhythm'] = df['filename'].apply(lambda x: x.split('_')[2])
    return df
df = split_metadata(df)
display(df.head())

#主成分分析に入れない列名を指定する
exclude_columns = ['filename', 'harmony', 'melody', 'rhythm']

#主成分分析に入れる列を選択する
feature_columns = [col for col in df.columns if col not in exclude_columns]
df_num = df[feature_columns]
display(df_num.info())

#特徴量を標準化する
scaler = StandardScaler()
df_num = scaler.fit_transform(df_num)
df_num = pd.DataFrame(df_num, columns=feature_columns)

# 全体特徴量をCSVに保存
output_path_whole = BASE_DIR / "acoustic_features_scaler.csv"
df_num.to_csv(output_path_whole, index=False, encoding="utf-8-sig")

#主成分分析を実行する
n_pca = 5
pca = PCA(n_components=n_pca)
pca_result = pca.fit_transform(df_num)

#主成分分析の結果をデータフレームに変換する
pca_columns = [f'PC{i+1}' for i in range(n_pca)]
df_pca = pd.DataFrame(pca_result, columns=pca_columns)

#主成分分析の結果を結合する
df_final = pd.concat([df[exclude_columns], df_pca], axis=1)

#主成分分析の累積寄与率を図示
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_pca + 1), np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

#主成分分析の各特徴量の寄与をヒートマップで図示する
# PCA loadingsをDataFrame化
loadings_df = pd.DataFrame(
    pca.components_.T,
    index=feature_columns,
    columns=pca_columns
)
plt.figure(figsize=(8, 14))
sns.heatmap(
    loadings_df,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Loading"}
)
plt.title("PCA Component Loadings", fontsize=14)
plt.xlabel("PCA Components", fontsize=12)
plt.ylabel("Original Features", fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()


#主成分分析の結果を図示するPC1,PC2をプロットする
sns.scatterplot(data=df_final, x='PC1', y='PC2', hue='harmony')
plt.title('PCA of Acoustic Features')
plt.show()

#PC1,PC3をプロットする
sns.scatterplot(data=df_final, x='PC1', y='PC3', hue='harmony')
plt.title('PCA of Acoustic Features')
plt.show()

#主成分分析の結果を図示するPC1,PC2をプロットする
sns.scatterplot(data=df_final, x='PC1', y='PC2', hue='melody')
plt.title('PCA of Acoustic Features')
plt.show()

#PC1,PC3をプロットする
sns.scatterplot(data=df_final, x='PC1', y='PC3', hue='melody')
plt.title('PCA of Acoustic Features')
plt.show()

#主成分分析の結果を図示するPC1,PC2をプロットする
sns.scatterplot(data=df_final, x='PC1', y='PC2', hue='rhythm')
plt.title('PCA of Acoustic Features')
plt.show()

#PC1,PC3をプロットする
sns.scatterplot(data=df_final, x='PC1', y='PC3', hue='rhythm')
plt.title('PCA of Acoustic Features')
plt.show()

#クラスタリングを実装する
from sklearn.cluster import KMeans
n_clus = 6
kmeans = KMeans(n_clusters=n_clus, random_state=0)
df_final['cluster'] = kmeans.fit_predict(df_pca)

#クラスタリングの結果を図示するPC1,PC2をプロットする
sns.scatterplot(data=df_final, x='PC1', y='PC2', hue='cluster', palette='Set1')
plt.title('KMeans Clustering of Acoustic Features')
plt.show()

#クラスタリングの結果を図示するPC1,PC3をプロットする
sns.scatterplot(data=df_final, x='PC1', y='PC3', hue='cluster', palette='Set1')
plt.title('KMeans Clustering of Acoustic Features')
plt.show()

#各クラスタの平均
cluster_centers = df_final.groupby("cluster")[pca_columns].mean()

#距離計算する関数
def calc_distance_to_center(row):
    cluster_id = row["cluster"]
    center = cluster_centers.loc[cluster_id].values
    point = row[pca_columns].values
    return np.linalg.norm(point - center)


# 各楽曲について、自分のクラスタ中心からの距離を計算
df_final["distance_to_cluster_center"] = df_final.apply(calc_distance_to_center, axis=1)

# 各クラスタで中心に最も近い楽曲を1曲ずつ抽出
representative_songs = (
    df_final
    .sort_values(["cluster", "distance_to_cluster_center"])
    .groupby("cluster")
    .head(1)
    .reset_index(drop=True)
)

print("代表曲のインデックス:")
print(representative_songs[['filename', 'cluster', 'distance_to_cluster_center']])



# 各クラスタごとの距離の平均
df_final["cluster_distance_mean"] = (
    df_final.groupby("cluster")["distance_to_cluster_center"]
    .transform("mean")
)

# 各クラスタごとの距離の標準偏差
df_final["cluster_distance_std"] = (
    df_final.groupby("cluster")["distance_to_cluster_center"]
    .transform("std")
)


# 平均 + 1標準偏差の距離
df_final["target_distance_1sd"] = (
    df_final["cluster_distance_mean"] + df_final["cluster_distance_std"]
)

# 各楽曲が「平均+1SDの距離」からどれだけ離れているか
df_final["diff_from_1sd"] = abs(
    df_final["distance_to_cluster_center"] - df_final["target_distance_1sd"]
)

# 各クラスタで、平均+1SDの距離に最も近い楽曲を1曲ずつ抽出
sd_songs = (
    df_final
    .sort_values(["cluster", "diff_from_1sd"])
    .groupby("cluster")
    .head(2)
    .reset_index(drop=True)
)

print("平均+1SD付近の楽曲:")
print(sd_songs[[
    "filename",
    "cluster",
    "distance_to_cluster_center",
    "target_distance_1sd",
    "diff_from_1sd"
]])

# 平均 + 2標準偏差の距離
df_final["target_distance_2sd"] = (
    df_final["cluster_distance_mean"] + 2 * df_final["cluster_distance_std"]
)

# 各楽曲が「平均+2SDの距離」からどれだけ離れているか
df_final["diff_from_2sd"] = abs(
    df_final["distance_to_cluster_center"] - df_final["target_distance_2sd"]
)

# 各クラスタで、平均+2SDの距離に最も近い楽曲を1曲ずつ抽出
sd_songs = (
    df_final
    .sort_values(["cluster", "diff_from_2sd"])
    .groupby("cluster")
    .head(2)
    .reset_index(drop=True)
)

print("平均+2SD付近の楽曲:")
print(sd_songs[[
    "filename",
    "cluster",
    "distance_to_cluster_center",
    "target_distance_2sd",
    "diff_from_2sd"
]])