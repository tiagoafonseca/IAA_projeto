import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --------------------- Carregamento e Preparação dos Dados ---------------------
# Carregar os dados
file_path = r'C:\Users\Tiago Afonseca\OneDrive - ISCTE-IUL\Documents\1º Year MEI\1º Semestre\IAA\projeto\CVD_cleaned_tester.csv'
data = pd.read_csv(file_path)

# Separar colunas numéricas e categóricas
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Introduzir valores ausentes artificiais (10%)
data_with_missing_10 = data.copy()
for col in data_with_missing_10.columns:
    if data_with_missing_10[col].dtype != 'object':
        data_with_missing_10.loc[data_with_missing_10.sample(frac=0.1).index, col] = np.nan

# Imputação: Média para numéricos, Moda para categóricos
imputer_numeric = SimpleImputer(strategy='mean')
imputer_categorical = SimpleImputer(strategy='most_frequent')

data_with_missing_10[numeric_cols] = imputer_numeric.fit_transform(data_with_missing_10[numeric_cols])
data_with_missing_10[categorical_cols] = imputer_categorical.fit_transform(data_with_missing_10[categorical_cols])

# Normalização das colunas numéricas
scaler = StandardScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data_with_missing_10[numeric_cols]), columns=numeric_cols)
data_normalized[categorical_cols] = data_with_missing_10[categorical_cols]

# Codificação de variáveis categóricas
data_encoded = pd.get_dummies(data_normalized, columns=categorical_cols, drop_first=True)

# Dados normalizados e preparados para k-Means
X_unsupervised = data_encoded

# --------------------- Experimento com k-Means ---------------------
# Determinando o número ideal de clusters (método do cotovelo)
inertia = []
silhouette_scores = []
cluster_range = range(2, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_unsupervised)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_unsupervised, kmeans.labels_))

# Plotando o método do cotovelo
plt.figure(figsize=(10, 5))
plt.plot(cluster_range, inertia, marker='o', label='Inércia')
plt.title("Método do Cotovelo (k-Means)", fontsize=14)
plt.xlabel("Número de Clusters (k)", fontsize=12)
plt.ylabel("Inércia", fontsize=12)
plt.grid(alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# Melhor número de clusters baseado no método da silhueta
best_k = cluster_range[silhouette_scores.index(max(silhouette_scores))]
print(f"Melhor número de clusters (baseado na Silhueta): {best_k}")

# Aplicando k-Means com o melhor número de clusters
kmeans_final = KMeans(n_clusters=best_k, random_state=42)
kmeans_labels = kmeans_final.fit_predict(X_unsupervised)

# --------------------- Análise dos Atributos Mais Relevantes ---------------------
# Extraindo os centroides dos clusters
centroids = pd.DataFrame(kmeans_final.cluster_centers_, columns=X_unsupervised.columns)

# Calculando a diferença média de cada atributo entre os clusters
centroid_differences = centroids.std().sort_values(ascending=False)

# Plotando os atributos mais relevantes com base na variação dos centroides
plt.figure(figsize=(10, 8))
centroid_differences.head(10).plot(kind='barh', color='salmon')
plt.title("Atributos mais Relevantes para os Clusters (k-Means)", fontsize=14)
plt.xlabel("Variação entre os Clusters", fontsize=12)
plt.ylabel("Atributos", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --------------------- Visualização dos Clusters ---------------------
# Reduzindo para duas dimensões para visualização (se necessário)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_unsupervised)

# Gráfico de dispersão dos clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.7)
plt.title("Clusters do k-Means (Reduzido para 2D)", fontsize=14)
plt.xlabel("Componente Principal 1", fontsize=12)
plt.ylabel("Componente Principal 2", fontsize=12)
plt.grid(alpha=0.5)
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()
