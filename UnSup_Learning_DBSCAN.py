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
import tables_class
from imblearn.over_sampling import SMOTE


# Carregar os dados
file_path = r'C:\Users\Tiago Afonseca\OneDrive - ISCTE-IUL\Documents\1º Year MEI\1º Semestre\IAA\projeto\CVD_cleaned_tester.csv'
data = pd.read_csv(file_path)

# --------------------- Preparação dos Dados ---------------------------
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

# --------------------- Normalização dos Dados ------------------------
# Normalização
scaler = StandardScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data_with_missing_10[numeric_cols]), columns=numeric_cols)
data_normalized[categorical_cols] = data_with_missing_10[categorical_cols]

# --------------------- Discretização dos Dados ------------------------
# Codificação de variáveis categóricas
data_encoded = pd.get_dummies(data_normalized, columns=categorical_cols, drop_first=True)

# Ajustando X e y
target_col = [col for col in data_encoded.columns if 'Heart_Disease' in col][0]
X = data_encoded.drop(columns=[target_col])
y = data_encoded[target_col]

# ---------------------- Aplicar SMOTE nos dados ----------------------
print("\nAntes do SMOTE:")
print(y.value_counts())

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

print("\nApós o SMOTE:")
print(pd.Series(y_smote).value_counts())

# -------------------- Redução de Dimensionalidade com PCA --------------------
n_atributos = 7  # Número de componentes principais
pca = PCA(n_components=n_atributos)

# Aplicar PCA nos dados normalizados (sem a variável target)
X_pca = pca.fit_transform(X_smote)

# Criar um DataFrame para as componentes principais
X_pca_df = pd.DataFrame(X_pca, columns=[f"PCA{i+1}" for i in range(n_atributos)])
X_pca_df['Heart_Disease'] = y_smote

# Substituir X pelos dados reduzidos
X_unsupervised = X_pca_df.drop(columns=['Heart_Disease'])

tables_class.creating_table(X_unsupervised)  # Visualizar tabela reduzida

# -------------------- Adicionando a Análise de Contribuições de cada Atributo(Loadings) --------------------
# Obter os loadings do PCA
loadings = pca.components_

# Criar um DataFrame para relacionar os loadings às variáveis originais
loading_matrix = pd.DataFrame(loadings, columns=X.columns, index=[f"PCA{i+1}" for i in range(n_atributos)])
#data_preparation.creating_table(loading_matrix)

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

# --------------------- GridSearch para DBSCAN ---------------------
from itertools import product  # Para gerar combinações de parâmetros

# Configuração de ranges para eps e min_samples
eps_values = np.linspace(0.1, 5.0, 20)  # Intervalo de eps
min_samples_values = range(2, 10)       # Intervalo de min_samples

# Inicializar variáveis para armazenar os melhores resultados
best_eps = None
best_min_samples = None
best_silhouette = -1
best_labels = None

# Loop para encontrar a melhor combinação de eps e min_samples
for eps, min_samples in product(eps_values, min_samples_values):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_unsupervised)

    # Avaliar as métricas
    if len(set(labels)) > 1:  # Garantir que há mais de um cluster
        sil_score = silhouette_score(X_unsupervised, labels)
        if sil_score > best_silhouette:  # Atualizar os melhores parâmetros
            best_eps = eps
            best_min_samples = min_samples
            best_silhouette = sil_score
            best_labels = labels

# Exibir os melhores resultados
print(f"\nMelhor valor de eps: {best_eps}")
print(f"Melhor valor de min_samples: {best_min_samples}")
print(f"Melhor Coeficiente de Silhueta: {best_silhouette:.2f}")

# Aplicar o DBSCAN com os melhores parâmetros
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
final_labels = dbscan.fit_predict(X_unsupervised)

# --------------------- Visualização dos Resultados ---------------------
n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
n_outliers = list(final_labels).count(-1)

print(f"\nNúmero de clusters: {n_clusters}")
print(f"Número de outliers: {n_outliers}")

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
colors = ["red" if label == -1 else "blue" if label == 0 else "green" for label in final_labels]

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
print("\nMétricas de Avaliação")
# Coeficiente de Silhueta
silhouette_avg = silhouette_score(X_unsupervised, final_labels)
print(f"Coeficiente de Silhueta: {silhouette_avg:.2f}")

# Índice de Calinski-Harabasz
ch_score = calinski_harabasz_score(X_unsupervised, final_labels)
print(f"Índice de Calinski-Harabasz: {ch_score:.2f}")

# Índice de Davies-Bouldin
db_score = davies_bouldin_score(X_unsupervised, final_labels)
print(f"Índice de Davies-Bouldin: {db_score:.2f}")

# --------------------- Resumo dos Clusters ---------------------
# Contagem de pontos por cluster
cluster_counts = pd.Series(final_labels).value_counts()
print("\nDistribuição dos Clusters:")
print(cluster_counts)
