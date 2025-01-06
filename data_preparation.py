import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import plotly.graph_objects as go



def creating_table(data):
    df = pd.DataFrame(data)

    # Criar tabela interativa com Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='cadetblue',
                    align='center',
                    font=dict(size=16, color='black'),
                    line_color='black'),

        # Aplicar estilo de negrito às células alteradas
        cells=dict(values=[df[col] for col in df.columns],
                   fill_color='white',
                   align='center',
                   font=dict(size=14, color='black'),
                   line_color='black'),

    )])

    # Calcular a largura da coluna com base no tamanho maximo do conteúdo das linhas da coluna e do cabeçalho
    column_widths = [
        max(len(column), max(len(str(value)) for value in df[column].astype(str))) * 10 + 20
        for column in df.columns
    ]

    # Ajustar a largura das colunas
    fig.data[0].columnwidth = column_widths

    # Exibir a tabela com rolagem automática
    fig.update_layout(
        width=3000,  # Largura total da tabela
        height=850,  # Altura da tabela
        margin=dict(l=10, r=10, t=10, b=10),
    )

    fig.show()

def creating_table_with_style(data, style_map):
    df = pd.DataFrame(data)

    # Criar tabela interativa com Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='cadetblue',
                    align='center',
                    font=dict(size=16, color='black'),
                    line_color='black'),

        # Aplicar estilo de negrito às células alteradas
        cells=dict(values=[df[col] for col in df.columns],
                   fill_color='white',
                   align='center',
                   font=dict(size=14, color=[['black' if style == '' else 'red' for style in style_map[col]] for col in df.columns]),
                   line_color='black'),

    )])


    # Calcular a largura da coluna com base no tamanho do cabeçalho
    column_widths = [len(column) * 10 + 20 for column in df.columns]

    # Ajustar a largura das colunas
    fig.data[0].columnwidth = column_widths

    # Exibir a tabela com rolagem automática
    fig.update_layout(
        width=2400,  # Largura total da tabela
        height=800,  # Altura da tabela
        margin=dict(l=10, r=10, t=10, b=10),
    )

    fig.show()


# Handdling Missing Values
# Função para remover aleatoriamente e substituir uma percentagem de valores
def randomly_remove_and_substitute_values(data, percentage, estrategia):
    data_with_missing = data.copy()
    total_values = data.size
    num_missing = int(total_values * (percentage / 100))
    substitute_value = 0

    # Gerar pares únicos de índices para substituições
    unique_indices = set()
    while len(unique_indices) < num_missing:
        row = np.random.choice(data_with_missing.index)
        col = np.random.choice(data_with_missing.columns)
        unique_indices.add((row, col))  # Adicionar pares únicos

    # Criar um mapa de estilos para sinalizar valores alterados
    style_map = pd.DataFrame('', index=data.index, columns=data.columns)

    # Substituir os valores selecionados pelo valor desejado
    for row, col in unique_indices:
        if type(data_with_missing.at[row, col]) in (int, float, complex) and estrategia == "dummy":
            data_with_missing.at[row, col] = 0
            style_map.at[row, col] = 'bold'
        elif type(data_with_missing.at[row, col]) not in (int, float, complex) and estrategia == "dummy":
            # Converter a coluna para aceitar strings
            data_with_missing[col] = data_with_missing[col].astype('object')
            data_with_missing.at[row, col] = "unknow"
            style_map.at[row, col] = 'bold'

        elif type(data_with_missing.at[row, col]) in (int, float, complex) and estrategia == "valor medio":
            data_with_missing.at[row, col] = data[col].mean()
            style_map.at[row, col] = 'bold'
        elif type(data_with_missing.at[row, col]) not in (int, float, complex) and estrategia == "valor medio":
            data_with_missing.at[row, col] = data[col].mode()[0]  # [0] acessa o valor mais frequente
            style_map.at[row, col] = 'bold'

    return data_with_missing, style_map



# Data Normalization - Normalização das variáveis contínuas
def data_normalization(data):
    continuous_vars = ['Height_(cm)', 'Weight_(kg)', 'Alcohol_Consumption', 'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption']
    scaler = MinMaxScaler()
    data[continuous_vars] = scaler.fit_transform(data[continuous_vars])
    data[continuous_vars] = round(data[continuous_vars], 3)  # Round to the specified number of decimal places

    # Exemplo de Label Encoding para colunas binárias
    binary_columns = ['Exercise', 'Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Depression',
                      'Diabetes', 'Arthritis', 'Smoking_History']
    for col in binary_columns:
        data[col] = data[col].map({'Yes': 1, 'No': 0})


    return data


# Data Discretization - Codificação de variáveis categóricas
def data_discretization(data):
    bins_1 = [1, 18, 35, 64, 100]
    labels_1 = ['child', 'young adult', 'middle-aged', 'senior']
    data['Age_Numeric'] = data['Age_Category'].str.extract(r'(\d+)').astype(int)
    data['Age_Category'] = pd.cut(data['Age_Numeric'], bins=bins_1, labels=labels_1)  # str.extract(r'(\d+)') to extract the numeric part of the age range (e.g., "70-74" → 70).Converts the extracted value to an integer using .astype(int).

    bins_2 = [1, 18.5, 25, 29.9, 60]  # [0, 0.120, 0.333, 0.494, 1] # intervalo normalizado -> [1, 18.5, 25, 29.9, 60] intervalos em kg
    labels_2 = ['Peso Baixo', 'Peso Normal', 'Excesso de Peso', 'Obesidade']
    data['BMI'] = pd.cut(data['BMI'], bins=bins_2, labels=labels_2)
    # bins_2_1 = [1, 22, 27, 60]  -> para idosos
    # labels_2_1 = ['Peso Baixo', 'Peso Normal', 'Excesso de Peso']

    return data




# Data Reduction
def data_reduction(data):
    # Aggregation
    data['Eating_Habits'] = (data['Fruit_Consumption'] + data['Green_Vegetables_Consumption']) - (
                data['FriedPotato_Consumption'] + data['Alcohol_Consumption'])
    bins_3 = [-4, -0.5, 0.0, 0.5, 4]  # ALTERAR OU REVER
    labels_3 = ['Muito Pouco Saudável', 'Pouco Saudável', 'Saudável', 'Muito Saudável']
    data['Eating_Habits'] = pd.cut(data['Eating_Habits'], bins=bins_3, labels=labels_3)


    # Sampling - Lista de colunas a remover
    columns_to_remove = [
        "Age_Numeric",
        "Checkup",
        "Weight_(kg)",
        "Alcohol_Consumption",
        "Fruit_Consumption",
        "Green_Vegetables_Consumption",
        "FriedPotato_Consumption"
    ]

    for col in columns_to_remove:
        if col in data.columns:
            data = data.drop(columns=col)

    return data





# Testing
#file_path = r'C:\Users\Tiago Afonseca\OneDrive - ISCTE-IUL\Documents\1º Year MEI\1º Semestre\IAA\projeto\CVD_cleaned_tester.csv'   # Prefixe o caminho com um r para indicar uma ‘string’ "raw" (crua), que ignora caracteres de escape
#data = pd.read_csv(file_path)
#creating_table(data)  # Tabela Inicial

#missing_values, style_map = randomly_remove_and_substitute_values(data, 10, "valor medio")  # Estratégia 1: dummy; Estratégia 2: valor medio
#creating_table_with_style(missing_values, style_map)
#normalized = data_normalization(data)
#discret = data_discretization(data)
#reduction = data_reduction(data)
#creating_table(reduction)


