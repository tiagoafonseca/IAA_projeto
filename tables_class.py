import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# classe de criação de tabelas
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
