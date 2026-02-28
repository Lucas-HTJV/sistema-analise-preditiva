import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, fonte_dados):
        """
        fonte_dados: pode ser caminho do arquivo (str) ou objeto BytesIO (Colab)
        """
        self.fonte = fonte_dados
        self.df = None

    def carregar(self):
        """Carrega arquivo Excel"""
        try:
            self.df = pd.read_excel(self.fonte)
        except Exception as e:
            raise ValueError(f"Erro ao carregar arquivo: {e}")

        if self.df is None or self.df.empty:
            raise ValueError("Arquivo carregado está vazio.")

        return self.df

    def limpar(self, col_x, col_y):
        """Limpeza e preparação dos dados"""

        if self.df is None:
            raise ValueError("Dados não carregados. Execute carregar() primeiro.")

        # Criar cópia para evitar warnings
        df = self.df.copy()

        # Normalizar strings (remove espaços)
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Valores inválidos
        invalidos = ["Not Specified", "not specified", "NA", "N/A", "", " "]
        df.replace(invalidos, np.nan, inplace=True)

        # Verificar colunas
        for col in [col_x, col_y]:
            if col not in df.columns:
                raise ValueError(f"Coluna '{col}' não encontrada no dataset.")

        # Converter para numérico
        df[col_x] = pd.to_numeric(df[col_x], errors='coerce')
        df[col_y] = pd.to_numeric(df[col_y], errors='coerce')

        # Remover nulos
        df.dropna(subset=[col_x, col_y], inplace=True)

        if df.empty:
            raise ValueError("Após limpeza, não restaram dados válidos.")

        self.df = df
        return self.df

    def filtrar_owner(self):
        """Filtra dados por Owner (se existir)"""

        if self.df is None:
            raise ValueError("Dados não carregados.")

        if "Owner" not in self.df.columns:
            print("\nColuna 'Owner' não encontrada. Pulando filtro.")
            return self.df

        escolha = input("\nDeseja filtrar por Owner? (s/n): ").strip().lower()

        if escolha == 's':
            owners = self.df["Owner"].dropna().unique()
            print("\nOwners disponíveis:")
            for i, o in enumerate(owners):
                print(f"{i} - {o}")

            try:
                idx = int(input("Escolha o índice do Owner: "))
                owner_escolhido = owners[idx]
            except (ValueError, IndexError):
                print("Escolha inválida. Nenhum filtro aplicado.")
                return self.df

            self.df = self.df[self.df["Owner"] == owner_escolhido]

        return self.df