import pandas as pd

class UEVAnalyzer:
    """
    Classe para análise estatística de duas colunas numéricas de um DataFrame.
    """

    def __init__(self, dataframe, coluna_x, coluna_y):
        self.df = dataframe.copy()  # evita alterar o original
        self.coluna_x = coluna_x
        self.coluna_y = coluna_y

        self._validar_colunas()

    def _validar_colunas(self):
        """Verifica se as colunas existem no DataFrame"""
        for col in [self.coluna_x, self.coluna_y]:
            if col not in self.df.columns:
                raise ValueError(f"A coluna '{col}' não existe no DataFrame.")

    def resumo_estatistico(self, coluna):
        """
        Retorna estatísticas descritivas básicas de uma coluna.
        """
        serie = self.df[coluna].dropna()

        return {
            "quantidade": int(serie.count()),
            "minimo": float(serie.min()),
            "maximo": float(serie.max()),
            "media": float(serie.mean()),
            "mediana": float(serie.median())
        }

    def correlacao(self):
        """
        Calcula a correlação de Pearson entre as duas colunas.
        """
        dados_validos = self.df[[self.coluna_x, self.coluna_y]].dropna()
        return float(dados_validos[self.coluna_x].corr(dados_validos[self.coluna_y]))

    def calcular_razao_k(self):
        """
        Calcula a razão k = y / x e retorna estatísticas.
        """
        dados_validos = self.df[[self.coluna_x, self.coluna_y]].dropna().copy()

        # Evita divisão por zero
        dados_validos = dados_validos[dados_validos[self.coluna_x] != 0]

        dados_validos["k"] = dados_validos[self.coluna_y] / dados_validos[self.coluna_x]

        return {
            "minimo": float(dados_validos["k"].min()),
            "maximo": float(dados_validos["k"].max()),
            "mediana": float(dados_validos["k"].median())
        }

    def primeiras_linhas(self, n=10):
        """
        Retorna as primeiras linhas válidas do conjunto de dados.
        """
        return self.df[[self.coluna_x, self.coluna_y]].dropna().head(n)