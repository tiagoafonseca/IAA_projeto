from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import re
from sklearn.decomposition import PCA
import matplotlib
import data_preparation


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
target_col = [col for col in data_encoded.columns if 'Heart_Disease' in col][0]
X = data_encoded.drop(columns=[target_col])
y = data_encoded[target_col]

# Adicione aqui para verificar se y contém os valores esperados:
print("\nResumo da variável alvo (Heart Disease):")
print(y.value_counts())

X_unsupervised = data_encoded

# -------------------- Redução de Dimensionalidade com PCA --------------------
n_atributos = 4  # Número de componentes principais
pca = PCA(n_components=n_atributos)

# Aplicar PCA nos dados normalizados (sem a variável target)
X_pca = pca.fit_transform(data_encoded.drop(columns=[target_col]))

# Criar um DataFrame para as componentes principais
X_pca_df = pd.DataFrame(X_pca, columns=[f"PCA{i+1}" for i in range(n_atributos)])
X_pca_df['Heart_Disease'] = data_encoded[target_col].values

data_preparation.creating_table(X_pca_df)  # Visualizar tabela reduzida

# Verificar a explicação de variância acumulada
explained_variance = np.cumsum(pca.explained_variance_ratio_)
print(f"\nExplicação de variância acumulada para {n_atributos} componentes principais: {explained_variance[-1]:.2f}")

# -------------------- Adicionando a Análise de Contribuições de cada Atributo(Loadings) --------------------
# Obter os loadings do PCA
loadings = pca.components_

# Criar um DataFrame para relacionar os loadings às variáveis originais
loading_matrix = pd.DataFrame(loadings, columns=X.columns, index=[f"PCA{i+1}" for i in range(n_atributos)])
#data_preparation.creating_table(loading_matrix)

# Exibir a matriz de loadings completa
print("\nMatriz de Loadings (Contribuições):")
print(loading_matrix)

# Visualizar as contribuições de variáveis para cada componente principal
for i in range(n_atributos):  # Iterar sobre todas as PCAs
    plt.figure(figsize=(11, 7))
    loading_matrix.iloc[i].plot(kind='barh', color='mediumpurple', alpha=0.7)
    plt.title(f"Contribuições das Variáveis para o {loading_matrix.index[i]}", fontsize=16)
    plt.xlabel("Variáveis Originais", fontsize=14)
    plt.ylabel("Contribuição", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.show()

# Substituir X_unsupervised pelos dados reduzidos do PCA (sem a variável target)
X_unsupervised = X_pca_df.drop(columns=['Heart_Disease'])
y_target = X_pca_df['Heart_Disease']

# --------------------- Determinar o Melhor eps ---------------------
neighbors = NearestNeighbors(n_neighbors=2)
neighbors_fit = neighbors.fit(X_unsupervised)
distances, indices = neighbors_fit.kneighbors(X_unsupervised)

# Ordenar as distâncias
distances = np.sort(distances[:, 1], axis=0)

# Calcular as diferenças para determinar o "cotovelo"
differences = np.diff(distances)
diff_diff = np.diff(differences)

# Encontrar o índice do ponto de maior curvatura (cotovelo)
elbow_index = np.argmax(diff_diff) + 1

# Determinar o melhor valor de eps
eps = max(distances[elbow_index], 0.4)  # Usar um limite mínimo de 0.4 (ou outro valor adequado)
print(f"Melhor valor de eps encontrado: {eps}")

# --------------------- Aplicando DBSCAN ---------------------
min_samples = 2  # Ajustar este valor conforme necessário
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X_unsupervised)

# Verificando o número de clusters encontrados
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_outliers = list(dbscan_labels).count(-1)

print(f"Número de clusters encontrados: {n_clusters}")
print(f"Número de outliers: {n_outliers}")

# Associar os clusters à variável-alvo
data_with_clusters = pd.DataFrame(X_unsupervised, columns=X_unsupervised.columns)
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

# --------------------- Visualização dos PCA's---------------------
# Visualizar todas as contribuições dos atributos em todos os PCA's
# Importância dos Componentes Principais (PCA)
explained_variance_ratio = pca.explained_variance_ratio_

# Criar o DataFrame com as informações
pca_importance = pd.DataFrame({"PCA": [f"PCA{i+1}" for i in range(len(explained_variance_ratio))],
                               "Explained Variance": explained_variance_ratio}).sort_values(by="Explained Variance", ascending=False)

# Criar o gráfico de barras horizontal
plt.figure(figsize=(12, 8))
plt.barh(pca_importance["PCA"], pca_importance["Explained Variance"], color="mediumpurple", alpha=0.7)
plt.title("Importância dos Componentes Principais (PCA)", fontsize=16)
plt.xlabel("Proporção de Variância Explicada", fontsize=14)
plt.ylabel("Componentes Principais", fontsize=14)
plt.tight_layout()
plt.gca().invert_yaxis()  # Inverte o eixo Y para mostrar os mais importantes no topo
plt.show()

# --------------------- Visualizações dos clusters ---------------------
# Reduzindo os dados para 2 dimensões usando PCA para visualização
pca = PCA(n_components=3)
X_2d = pca.fit_transform(X_unsupervised)

# Mapeamento de rótulos para legendas
cluster_labels = {-1: "Outliers", 0: "No Disease", 1: "Disease"}

# Adicionar cores personalizadas para cada cluster
colors = ["red" if label == -1 else "blue" if label == 0 else "green" for label in dbscan_labels]

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.6, edgecolor="k", s=50)
plt.title("Visualização dos Clusters Formados (DBSCAN)", fontsize=16)
plt.xlabel("Componente Principal 1", fontsize=12)
plt.ylabel("Componente Principal 2", fontsize=12)

# Criar uma legenda personalizada
handles = [
    plt.Line2D([0], [0], marker="o", color="w", label="Outliers", markerfacecolor="red", markersize=10),
    plt.Line2D([0], [0], marker="o", color="w", label="No Disease", markerfacecolor="blue", markersize=10),
    plt.Line2D([0], [0], marker="o", color="w", label="Disease", markerfacecolor="green", markersize=12),
]
plt.legend(handles=handles, title="Legenda", loc="best")
plt.tight_layout()
plt.show()

# --------------------- Métricas de Avaliação ---------------------
# Coeficiente de Silhueta
silhouette_avg = silhouette_score(X_unsupervised, dbscan_labels)
print(f"Coeficiente de Silhueta: {silhouette_avg:.2f}")

# Índice de Calinski-Harabasz
ch_score = calinski_harabasz_score(X_unsupervised, dbscan_labels)
print(f"Índice de Calinski-Harabasz: {ch_score:.2f}")

# Índice de Davies-Bouldin
db_score = davies_bouldin_score(X_unsupervised, dbscan_labels)
print(f"Índice de Davies-Bouldin: {db_score:.2f}")

# --------------------- Resumo dos Clusters ---------------------
# Contagem de pontos por cluster
cluster_counts = pd.Series(dbscan_labels).value_counts()
print("\nDistribuição dos Clusters:")
print(cluster_counts)

# Resumo da variável alvo (se aplicável)
if "Heart_Disease" in data_with_clusters.columns:
    cluster_analysis = data_with_clusters.groupby(dbscan_labels)['Heart_Disease'].mean()
    print("\nProporção de 'Heart Disease' por Cluster:")
    print(cluster_analysis)
