import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
import re


# Carregar os dados
file_path = r'C:\Users\Tiago Afonseca\OneDrive - ISCTE-IUL\Documents\1º Year MEI\1º Semestre\IAA\projeto\CVD_cleaned_tester.csv'  # Substitua pelo caminho correto do arquivo
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

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ----------------- Experimento com Multilayer Perceptron (com SMOTE) -----------------
# Aplicando SMOTE com ajuste de número de vizinhos
smote_adjusted = SMOTE(random_state=42, k_neighbors=2)
X_resampled, y_resampled = smote_adjusted.fit_resample(X_train, y_train)

# Modelo: Multilayer Perceptron
mlp_clf_resampled = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_clf_resampled.fit(X_resampled, y_resampled)

# Avaliação
y_pred_mlp_resampled = mlp_clf_resampled.predict(X_test)
classification_rep_mlp_resampled = classification_report(y_test, y_pred_mlp_resampled, zero_division=1)
accuracy_mlp_resampled = accuracy_score(y_test, y_pred_mlp_resampled)

# Estimando a importância das características
perm_importance = permutation_importance(mlp_clf_resampled, X_test, y_test, n_repeats=10, random_state=42)

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


# Resultados
print("\nResultados do Multilayer Perceptron com SMOTE ajustado:")
print(classification_rep_mlp_resampled)
print(f"Acurácia: {accuracy_mlp_resampled:.2f}")

