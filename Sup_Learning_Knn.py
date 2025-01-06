import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import re
import data_preparation
from imblearn.over_sampling import SMOTE

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
target_col = [col for col in data_encoded.columns if 'Heart_Disease' in col][0]  # Detectando a coluna-alvo
X = data_encoded.drop(columns=[target_col])
y = data_encoded[target_col]


# -------------------- Redução de Dimensionalidade com PCA --------------------
# Configurando o PCA para reduzir para N componentes principais
n_atributos = 7
pca = PCA(n_components=n_atributos)

# Aplicando o PCA aos dados normalizados (X)
X_pca = pca.fit_transform(data_encoded.drop(columns=[target_col]))

# Criando um DataFrame com as N componentes principais
X_pca_df = pd.DataFrame(X_pca, columns=[f"PCA{i+1}" for i in range(n_atributos)])
X_pca_df['Heart_Disease'] = data_encoded[target_col]  # Adicionando a variável alvo

data_preparation.creating_table(X_pca_df)  # Visualizar tabela reduzida

# Exibir a explicação da variância acumulada
explained_variance = np.cumsum(pca.explained_variance_ratio_)
print(f"\nExplicação de variância acumulada para {n_atributos} componentes:", explained_variance[-1])

# -------------------- Adicionando a Análise de Contribuições de cada Atributo(Loadings) --------------------
# Obter os loadings do PCA
loadings = pca.components_

# Criar um DataFrame para relacionar os loadings às variáveis originais
loading_matrix = pd.DataFrame(loadings, columns=X.columns, index=[f"PCA{i+1}" for i in range(n_atributos)])

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

# Atualizar X e y com os dados reduzidos
X = X_pca_df.drop(columns=['Heart_Disease'])
y = X_pca_df['Heart_Disease']

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicar SMOTE apenas nos dados de treino
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Verificar o balanceamento após o SMOTE
print("Distribuição antes do SMOTE:", y_train.value_counts())
print("Distribuição após o SMOTE:", pd.Series(y_train_smote).value_counts())

# ----------------- Experimento com k-NN -----------------
# Modelo: k-NN
knn_clf = KNeighborsClassifier(n_neighbors=5)  # Utilizando 5 vizinhos como padrão

# Treinando o modelo
knn_clf.fit(X_train_smote, y_train_smote)

# Avaliação
y_pred_knn = knn_clf.predict(X_test)

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

# Outras métricas
classification_rep = classification_report(y_test, y_pred_knn)
accuracy = accuracy_score(y_test, y_pred_knn)
recall_macro = recall_score(y_test, y_pred_knn, average='macro')
recall_micro = recall_score(y_test, y_pred_knn, average='micro')
precision_macro = precision_score(y_test, y_pred_knn, average='macro')
precision_micro = precision_score(y_test, y_pred_knn, average='micro')
f1_macro = f1_score(y_test, y_pred_knn, average='macro')
f1_micro = f1_score(y_test, y_pred_knn, average='micro')


# Estimando a importância das características para k-NN usando Permutation Importance
perm_importance_knn = permutation_importance(knn_clf, X_test, y_test, n_repeats=10, random_state=42)

# Organizando os dados para visualização
feature_importance_knn = pd.Series(perm_importance_knn.importances_mean, index=X.columns)

# Ordenando os atributos pela importância
feature_importance_knn = feature_importance_knn.abs().sort_values(ascending=True)

# Plotando a importância das características
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_knn.index, feature_importance_knn.values, color='lightcoral')
plt.title("Importância dos Atributos - k-NN", fontsize=16)
plt.xlabel("Relevância na Previsão de Doença Cardíaca", fontsize=14)
plt.ylabel("Atributos", fontsize=14)
plt.tight_layout()
plt.show()


# Resultados
print("\nRelatório de Classificação:")
print(classification_rep)
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall (Macro): {recall_macro:.2f}")
print(f"Recall (Micro): {recall_micro:.2f}")
print(f"Precisão (Macro): {precision_macro:.2f}")
print(f"Precisão (Micro): {precision_micro:.2f}")
print(f"F1-Score (Macro): {f1_macro:.2f}")
print(f"F1-Score (Micro): {f1_micro:.2f}")
