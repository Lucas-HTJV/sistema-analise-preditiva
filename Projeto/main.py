"""
Arquivo principal para execução via terminal.
Responsável por orquestrar todo o fluxo da aplicação.
"""

# ================================
# AJUSTE DE PATH (resolve imports)
# ================================
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# ================================
# IMPORTS
# ================================
from src.data_loader import DataLoader
from src.analyzer import UEVAnalyzer
from src.models import RegressionModel, LogLogRegressionModel
from src.visualizer import Visualizer
from src.utils import selecionar_arquivo


# ================================
# ESCOLHER COLUNA
# ================================
def escolher_coluna(df, nome):
    print(f"\nSelecione a coluna para {nome}:")

    for i, col in enumerate(df.columns):
        print(f"{i} - {col}")

    while True:
        try:
            idx = int(input(f"Digite o número da coluna {nome}: "))
            if 0 <= idx < len(df.columns):
                return df.columns[idx]
            else:
                print("Número fora do intervalo.")
        except ValueError:
            print("Entrada inválida. Digite um número.")


# ================================
# MAIN
# ================================
def main():
    print("=== Sistema de Análise Estatística ===\n")

    caminho = selecionar_arquivo()

    if not caminho:
        print("Nenhum arquivo selecionado.")
        return

    # ----------------------------
    # Carregar dados
    # ----------------------------
    try:
        loader = DataLoader(caminho)
        df = loader.carregar()
    except Exception as e:
        print("Erro ao carregar arquivo:", e)
        return

    if df is None or df.empty:
        print("Arquivo vazio ou inválido.")
        return

    # ----------------------------
    # Escolher colunas
    # ----------------------------
    col_x = escolher_coluna(df, "X")
    col_y = escolher_coluna(df, "Y")

    # ----------------------------
    # Limpeza
    # ----------------------------
    df = loader.limpar(col_x, col_y)

    try:
        df = loader.filtrar_owner()
    except Exception:
        pass

    print(f"\nRegistros válidos: {len(df)}")

    # ----------------------------
    # Análise
    # ----------------------------
    analyzer = UEVAnalyzer(df, col_x, col_y)

    print("\n=== Estatísticas ===")
    print("X:", analyzer.resumo_estatistico(col_x))
    print("Y:", analyzer.resumo_estatistico(col_y))

    print("\n=== Correlação ===")
    print("Pearson:", analyzer.correlacao())

    print("\n=== Razão k ===")
    print(analyzer.calcular_razao_k())

    print("\n=== Primeiras 10 linhas ===")
    print(analyzer.primeiras_linhas())

    # ----------------------------
    # Modelos
    # ----------------------------
    print("\n=== Regressão Linear ===")
    try:
        reg = RegressionModel(df, col_x, col_y)
        print(reg.treinar())
    except Exception as e:
        print("Erro na regressão linear:", e)

    print("\n=== Regressão Log-Log ===")
    try:
        log_model = LogLogRegressionModel(df, col_x, col_y)
        print(log_model.treinar())
    except Exception as e:
        print("Erro na regressão log-log:", e)

    # ----------------------------
    # Visualização
    # ----------------------------
    try:
        viz = Visualizer(df, col_x, col_y)
        viz.plotar()
    except Exception as e:
        print("Erro ao gerar gráficos:", e)


# ================================
# EXECUÇÃO
# ================================
if __name__ == "__main__":
    main()