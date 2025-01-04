import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA

# Carregar os dados
file_path = r'C:\Users\Tiago Afonseca\OneDrive - ISCTE-IUL\Documents\1º Year MEI\1º Semestre\IAA\projeto\CVD_cleaned_tester.csv'
data = pd.read_csv(file_path)

# Inspeção inicial
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Remoção de Duplicados
data = data.drop_duplicates()

# Introduzir valores ausentes artificiais (10% ou 20%)
# Solicitar a percentagem de valores omissos
while True:
    percentagem = input("Introduza a percentagem de valores omissos (10% ou 20%): ").strip()
    percentagem = re.sub(r"[^\w\s]", "", percentagem)  # Remove caracteres não numéricos
    if percentagem in {"10", "20"}:  # Verifica se a entrada é válida
        break
    else:
        print("Entrada inválida! Por favor, introduza '10%' ou '20%'.")

# Criar cópia do dataset e introduzir valores omissos
data_with_missing_10 = data.copy()
frac = 0.1 if percentagem == "10" else 0.2

for col in data_with_missing_10.columns:
    if data_with_missing_10[col].dtype != 'object':  # Apenas colunas numéricas
        data_with_missing_10.loc[data_with_missing_10.sample(frac=frac).index, col] = np.nan


# Tratamento de valores omissos
while True:
    estrategia = input("Introduza a estratégia de tratamento de valores omissos! "
                       "Estratégia 1 - Média/Moda; Estratégia 2 - Remoção da linha com valor omisso: ").strip()
    if estrategia in {"1", "2"}:  # Verifica se a entrada é válida
        break
    else:
        print("Entrada inválida! Por favor, introduza '1' para Média/Moda ou '2' para Remoção da linha.")

if int(estrategia) == 1:
    # Estratégia de Substituição: Média para numéricos, Moda para categóricos
    imputer_numeric = SimpleImputer(strategy='mean')
    imputer_categorical = SimpleImputer(strategy='most_frequent')

    data_with_missing_10[numeric_cols] = imputer_numeric.fit_transform(data_with_missing_10[numeric_cols])
    data_with_missing_10[categorical_cols] = imputer_categorical.fit_transform(data_with_missing_10[categorical_cols])
else:
    # Estratégia de Remoção: Remoção da linha completa de conter um null
    data_with_missing_10 = data_with_missing_10.dropna()


# Normalização
scaler = StandardScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data_with_missing_10[numeric_cols]), columns=numeric_cols)
data_normalized[categorical_cols] = data_with_missing_10[categorical_cols]

# Codificação de variáveis categóricas
data_encoded = pd.get_dummies(data_normalized, columns=categorical_cols, drop_first=True)

# Ajustando X e y
target_col = [col for col in data_encoded.columns if 'Heart' in col][0]  # Detectando a coluna-alvo
X = data_encoded.drop(columns=[target_col])
y = data_encoded[target_col]

# Codificação de variáveis categóricas
data_encoded = pd.get_dummies(data_normalized, columns=categorical_cols, drop_first=True)

# Dados normalizados e preparados para k-Means
X_unsupervised = data_encoded

# Determinando o número ideal de ‘clusters’ (método do cotovelo)
inertia = []
silhouette_scores = []
cluster_range = range(2, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_unsupervised)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_unsupervised, kmeans.labels_))

# Plot do método do cotovelo
plt.figure(figsize=(10, 5))
plt.plot(cluster_range, inertia, marker='o', label='Inércia')
plt.title("Método do Cotovelo (k-Means)", fontsize=14)
plt.xlabel("Número de Clusters (k)", fontsize=12)
plt.ylabel("Inércia", fontsize=12)
plt.grid(alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# Melhor número de ‘clusters’ baseado no método da silhueta
best_k = cluster_range[silhouette_scores.index(max(silhouette_scores))]
print(f"Melhor número de clusters (baseado na Silhueta): {best_k}")

# Aplicando k-Means com o melhor número de ‘clusters’
kmeans_final = KMeans(n_clusters=best_k, random_state=42)
kmeans_labels = kmeans_final.fit_predict(X_unsupervised)

# --------------------- Análise dos Atributos Mais Relevantes ---------------------
# Extraindo os centroides dos ‘clusters’
centroids = pd.DataFrame(kmeans_final.cluster_centers_, columns=X_unsupervised.columns)

# Calculando a diferença média de cada atributo entre os ‘clusters’
centroid_differences = centroids.std().sort_values(ascending=False)

# Plotando os atributos mais relevantes com base na variação dos centroides
plt.figure(figsize=(12, 8))
centroid_differences.head(10).plot(kind='barh', color='salmon')
plt.title("Importância dos Atributos - Decision Tree", fontsize=16)
plt.xlabel("Variação entre os Clusters", fontsize=14)
plt.ylabel("Atributos", fontsize=14)
plt.tight_layout()
plt.show()

# --------------------- Visualização dos ‘Clusters’ ---------------------
# Reduzindo para duas dimensões para visualização (se necessário)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_unsupervised)

# Gráfico de dispersão dos ‘clusters’
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.7)
plt.title("Clusters do k-Means (Reduzido para 2D)", fontsize=16)
plt.xlabel("Componente Principal 1", fontsize=14)
plt.ylabel("Componente Principal 2", fontsize=14)
plt.colorbar(scatter, label="Heart Disease (0=No, 1=Yes)")
plt.tight_layout()
plt.gca().invert_yaxis()  # Inverte o eixo Y para o maior atributo aparecer no topo
plt.show()
