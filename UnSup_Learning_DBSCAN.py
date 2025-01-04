from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import re
from sklearn.decomposition import PCA
import matplotlib

# Carregar os dados
file_path = r'C:\Users\Tiago Afonseca\OneDrive - ISCTE-IUL\Documents\1º Year MEI\1º Semestre\IAA\projeto\CVD_cleaned_tester.csv'
data = pd.read_csv(file_path)

# Inspeção inicial
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Remoção de Duplicados
data = data.drop_duplicates()

# Introduzir valores ausentes artificiais (10% ou 20%)
while True:
    percentagem = input("Introduza a percentagem de valores omissos (10% ou 20%): ").strip()
    percentagem = re.sub(r"[^\w\s]", "", percentagem)
    if percentagem in {"10", "20"}:
        break
    else:
        print("Entrada inválida! Por favor, introduza '10%' ou '20%'.")

data_with_missing_10 = data.copy()
frac = 0.1 if percentagem == "10" else 0.2

for col in data_with_missing_10.columns:
    if data_with_missing_10[col].dtype != 'object':
        data_with_missing_10.loc[data_with_missing_10.sample(frac=frac).index, col] = np.nan

# Tratamento de valores omissos
while True:
    estrategia = input("Introduza a estratégia de tratamento de valores omissos! "
                       "Estratégia 1 - Média/Moda; Estratégia 2 - Remoção da linha com valor omisso: ").strip()
    if estrategia in {"1", "2"}:
        break
    else:
        print("Entrada inválida! Por favor, introduza '1' para Média/Moda ou '2' para Remoção da linha.")

if int(estrategia) == 1:
    imputer_numeric = SimpleImputer(strategy='mean')
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    data_with_missing_10[numeric_cols] = imputer_numeric.fit_transform(data_with_missing_10[numeric_cols])
    data_with_missing_10[categorical_cols] = imputer_categorical.fit_transform(data_with_missing_10[categorical_cols])
else:
    data_with_missing_10 = data_with_missing_10.dropna()

# Normalização
scaler = StandardScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data_with_missing_10[numeric_cols]), columns=numeric_cols)
data_normalized[categorical_cols] = data_with_missing_10[categorical_cols]

# Codificação de variáveis categóricas
data_encoded = pd.get_dummies(data_normalized, columns=categorical_cols, drop_first=True)

# Ajustando X e y
target_col = [col for col in data_encoded.columns if 'Heart' in col][0]
X = data_encoded.drop(columns=[target_col])
y = data_encoded[target_col]

# Adicione aqui para verificar se y contém os valores esperados:
print("Resumo da variável alvo (Heart Disease):")
print(y.value_counts())

X_unsupervised = X.copy()

# --------------------- Determinando o Melhor eps ---------------------
neighbors = NearestNeighbors(n_neighbors=6)
neighbors_fit = neighbors.fit(X_unsupervised)
distances, indices = neighbors_fit.kneighbors(X_unsupervised)

# Gráfico para determinar o melhor valor de eps
distances = np.sort(distances[:, 5], axis=0)
plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.title("Gráfico para Determinação do eps (DBSCAN)", fontsize=14)
plt.xlabel("Amostras", fontsize=12)
plt.ylabel("Distância do 7º Vizinho Mais Próximo", fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

# --------------------- Aplicando DBSCAN ---------------------
eps = float(input("Escolha um valor para eps com base no gráfico: "))
min_samples = 7  # Pode ajustar este valor dependendo dos dados

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X_unsupervised)

# Verificando o número de clusters encontrados
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_outliers = list(dbscan_labels).count(-1)

print(f"Número de clusters encontrados: {n_clusters}")
print(f"Número de outliers: {n_outliers}")

# Associar os clusters à variável-alvo
data_with_clusters = pd.DataFrame(X, columns=X.columns)
data_with_clusters['Cluster'] = dbscan_labels
data_with_clusters['Heart Disease'] = y

# Proporção de Heart Disease em cada cluster
cluster_analysis = data_with_clusters.groupby('Cluster')['Heart Disease'].mean()
print("\nProporção de Heart Disease por Cluster:")
print(cluster_analysis)

# Análise descritiva para cada cluster
cluster_summary = data_with_clusters.groupby('Cluster').mean()
print("\nResumo Estatístico por Cluster:")
print(cluster_summary)

# --------------------- Visualizações ---------------------
# Visualização da Distribuição Real de Heart Disease
plt.figure(figsize=(8, 6))
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='coolwarm', alpha=0.6, edgecolor="k")
plt.title("Distribuição Real de Heart Disease", fontsize=14)
plt.xlabel("Componente Principal 1", fontsize=12)
plt.ylabel("Componente Principal 2", fontsize=12)
plt.colorbar(label="Heart Disease")
plt.tight_layout()
plt.show()


# --------------------- Resumo Estatístico dos Atributos ---------------------
cluster_summary = data_with_clusters.groupby('Cluster').mean()
attribute_variation = cluster_summary.std().sort_values(ascending=False)
top_attributes = attribute_variation.head(10).index
cluster_summary_top = cluster_summary[top_attributes]

plt.figure(figsize=(12, 8))
cluster_summary_top.T.plot(kind='bar', figsize=(12, 8), colormap='viridis', width=0.8)
plt.title("Resumo Estatístico dos Clusters - Atributos Mais Relevantes", fontsize=16)
plt.ylabel("Média dos Atributos", fontsize=14)
plt.xlabel("Atributos", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title="Clusters", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
