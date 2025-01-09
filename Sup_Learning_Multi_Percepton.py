import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc)
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.inspection import permutation_importance
import re
import tables_class

# Carregar os dados
file_path = r'C:\Users\Tiago Afonseca\OneDrive - ISCTE-IUL\Documents\1º Year MEI\1º Semestre\IAA\projeto\CVD_cleaned_tester.csv'  # Substitua pelo caminho correto do arquivo
data = pd.read_csv(file_path)

# --------------------- Preparação dos Dados ---------------------------
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

# --------------------- Normalização dos Dados --------------------------
# Normalização
scaler = StandardScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data_with_missing_10[numeric_cols]), columns=numeric_cols)
data_normalized[categorical_cols] = data_with_missing_10[categorical_cols]

# --------------------- Discretização dos Dados --------------------------
# Codificação de variáveis categóricas
data_encoded = pd.get_dummies(data_normalized, columns=categorical_cols, drop_first=True)

# Ajustando X e y
target_col = [col for col in data_encoded.columns if 'Heart_Disease' in col][0]  # Detectando a coluna-alvo
X = data_encoded.drop(columns=[target_col])
y = data_encoded[target_col]

# -------------------- Redução de Dimensionalidade com PCA --------------------
# Configurando o PCA para reduzir para N componentes principais
n_atributos = 10
pca = PCA(n_components=n_atributos)

# Aplicando o PCA aos dados normalizados (X)
X_pca = pca.fit_transform(data_encoded.drop(columns=[target_col]))

# Criando um DataFrame com as N componentes principais
X_pca_df = pd.DataFrame(X_pca, columns=[f"PCA{i+1}" for i in range(n_atributos)])
X_pca_df['Heart_Disease'] = data_encoded[target_col]  # Adicionando a variável alvo

tables_class.creating_table(X_pca_df)  # Visualizar tabela reduzida

# Exibir a explicação da variância acumulada
explained_variance = np.cumsum(pca.explained_variance_ratio_)
print(f"\nExplicação de variância acumulada para {n_atributos} componentes:", explained_variance[-1])

# -------------------- Adicionando a Análise de Contribuições de cada Atributo (Loadings) --------------------
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

# Atualizar X e y com os dados reduzidos
X = X_pca_df.drop(columns=['Heart_Disease'])
y = X_pca_df['Heart_Disease']

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ---------------------- Aplicar SMOTE apenas nos dados de treino ----------------------
smote = SMOTE(random_state=42, k_neighbors=2)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Verificar o balanceamento após o SMOTE
print("Distribuição antes do SMOTE:", y_train.value_counts())
print("Distribuição após o SMOTE:", pd.Series(y_train_smote).value_counts())

# ---------------------- Aplicar Random UnderSampling apenas nos dados de treino ----------------------
"""rus = RandomUnderSampler(random_state=42)
X_train_smote, y_train_smote = rus.fit_resample(X_train, y_train)

# Verificar o balanceamento após o undersampling
print("Distribuição antes do undersampling:", y_train.value_counts())
print("Distribuição após o undersampling:", pd.Series(y_train_smote).value_counts())"""

# ----------------- Ajuste de Hiperparâmetros para MLPClassifier -----------------
# Definir os hiperparâmetros para o GridSearch
param_grid = {
    'hidden_layer_sizes': [(100, 50)],  # Número de neurónios por camada
    'activation': ['relu', 'tanh'],  # Funções de ativação
    'max_iter': [2000]  # Número de iterações máximas
}

# Modelo: ML-Perceptron
mlp_clf = MLPClassifier()  # random_state=42 ???

# Configurar o GridSearchCV
grid_search = GridSearchCV(
    estimator=mlp_clf,
    param_grid=param_grid,
    scoring='f1_macro',  # Métrica para otimização
    cv=5,  # Validação cruzada com 5 folds
    verbose=1,  # Mostrar progresso do GridSearch
    n_jobs=-1  # Utilizar todos os núcleos disponíveis
)

# Ajustar o GridSearch nos dados balanceados (após SMOTE ou undersampling)
grid_search.fit(X_train_smote, y_train_smote)

# Exibir os melhores hiperparâmetros
print("Melhores hiperparâmetros encontrados: ", grid_search.best_params_)

# Melhor modelo ajustado
best_mlp_model = grid_search.best_estimator_

# Avaliação com o melhor modelo nos dados de teste
y_pred_mlp_resampled = best_mlp_model.predict(X_test)

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred_mlp_resampled)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

# Calcular a curva ROC e AUC
y_probs = best_mlp_model.predict_proba(X_test)[:, 1]  # Probabilidades da classe "Disease"
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

# Plotar a curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Linha diagonal
plt.title("Curva ROC", fontsize=16)
plt.xlabel("Taxa de Falsos Positivos (FPR)", fontsize=12)
plt.ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

# Calcular a curva Precision-Recall
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)

# Plotar a curva Precision-Recall
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
plt.title("Curva Precision-Recall", fontsize=16)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.legend(loc="lower left", fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

# Outras métricas
classification_rep = classification_report(y_test, y_pred_mlp_resampled)
accuracy = accuracy_score(y_test, y_pred_mlp_resampled)
recall_macro = recall_score(y_test, y_pred_mlp_resampled, average='macro')
recall_micro = recall_score(y_test, y_pred_mlp_resampled, average='micro')
precision_macro = precision_score(y_test, y_pred_mlp_resampled, average='macro')
precision_micro = precision_score(y_test, y_pred_mlp_resampled, average='micro')
f1_macro = f1_score(y_test, y_pred_mlp_resampled, average='macro')
f1_micro = f1_score(y_test, y_pred_mlp_resampled, average='micro')

# Estimando a importância das características
perm_importance = permutation_importance(best_mlp_model, X_test, y_test, n_repeats=10, random_state=42)

# Organizando os dados em um DataFrame para visualização
feature_importance = pd.Series(perm_importance.importances_mean, index=X.columns)

# Ajustando a escala para valores absolutos e ordenando para melhor visualização
feature_importance = feature_importance.abs().sort_values(ascending=True)


# Visualizando a importância
plt.figure(figsize=(12, 8))
plt.barh(feature_importance.index, feature_importance.values, color='darkkhaki')
plt.title("Importância dos Atributos - Mul.Perceptron", fontsize=16, )
plt.xlabel("Relevância na Previsão de Doença Cardíaca", fontsize=14)
plt.ylabel("Atributos", fontsize=14)
plt.tight_layout()
plt.show()


# ----------------- Exibir as Métricas --------------------
print("\nRelatório de Classificação:")
print(classification_rep)
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall (Macro): {recall_macro:.2f}")
print(f"Recall (Micro): {recall_micro:.2f}")
print(f"Precisão (Macro): {precision_macro:.2f}")
print(f"Precisão (Micro): {precision_micro:.2f}")
print(f"F1-Score (Macro): {f1_macro:.2f}")
print(f"F1-Score (Micro): {f1_micro:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")
print(f"Precision-Recall AUC Score: {pr_auc:.2f}")

