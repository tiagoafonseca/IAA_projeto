import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import re
import data_preparation


# Carregar os dados
file_path = r'C:\Users\Tiago Afonseca\OneDrive - ISCTE-IUL\Documents\1º Year MEI\1º Semestre\IAA\projeto\CVD_cleaned_tester_2.csv'
data = pd.read_csv(file_path)

# Inspeção inicial
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Criar um mapa de estilos para sinalizar valores alterados
style_map = pd.DataFrame('', index=data.index, columns=data.columns)

# Introduzir valores ausentes artificiais (10% ou 20%)
percentagem = input("Introduza a percentagem de valores omissos (10% ou 20%): ")
percentagem = re.sub(r"[^\w\s]", "", percentagem)  # remove a '%'
while percentagem == 10 or percentagem == 20:
    percentagem = input("Introduza a percentagem de valores omissos (10% ou 20%): ")
    percentagem = re.sub(r"[^\w\s]", "", percentagem)
data_with_missing_10 = data.copy()
for col in data_with_missing_10.columns:
    if data_with_missing_10[col].dtype != 'object':
        if percentagem == "10" or percentagem == "10%":
            data_with_missing_10.loc[data_with_missing_10.sample(frac=0.1).index, col] = np.nan
        else:
            data_with_missing_10.loc[data_with_missing_10.sample(frac=0.2).index, col] = np.nan



while True:
    estrategia = input("Introduza a estratégia de tratamento de valores omissos! "
                       "Estratégia 1 - Média/Moda; Estratégia 2 - Remoção da linha com valor omisso: ").strip()
    if estrategia in {"1", "2"}:  # Verifica se a entrada é válida
        break
    else:
        print("Entrada inválida! Por favor, introduza '1' para Média/Moda ou '2' para Remoção da linha.")

if int(estrategia) == 1:
    # Valores Omissos Estratégia de Substituição: Média para numéricos, Moda para categóricos
    imputer_numeric = SimpleImputer(strategy='mean')
    imputer_categorical = SimpleImputer(strategy='most_frequent')

    data_with_missing_10[numeric_cols] = imputer_numeric.fit_transform(data_with_missing_10[numeric_cols])
    data_with_missing_10[categorical_cols] = imputer_categorical.fit_transform(data_with_missing_10[categorical_cols])
    data_preparation.creating_table(data_with_missing_10)
else:
    data_with_missing_10 = data_with_missing_10.dropna()
    data_preparation.creating_table(data_with_missing_10)



