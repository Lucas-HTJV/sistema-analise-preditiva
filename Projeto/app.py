"""
Aplicação web para visualização dos dados.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def carregar_arquivo(arquivo):
    """
    Carrega arquivos em diferentes formatos.
    """
    try:
        if arquivo.name.endswith(".csv"):
            return pd.read_csv(arquivo)

        elif arquivo.name.endswith(".xlsx") or arquivo.name.endswith(".xls"):
            return pd.read_excel(arquivo)

        elif arquivo.name.endswith(".json"):
            return pd.read_json(arquivo)

        elif arquivo.name.endswith(".sql"):
            st.warning("Arquivos SQL ainda não são suportados diretamente.")
            return None

        else:
            st.error("Formato de arquivo não suportado.")
            return None

    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None


def main():
    """
    Interface web para análise de dados.
    """

    st.set_page_config(
        page_title="Dashboard de Dados",
        layout="wide"
    )

    st.title("📊 Dashboard de Análise de Dados")

    arquivo = st.file_uploader(
        "Selecione o arquivo",
        type=["xlsx", "csv", "sql", "json"]
    )

    if arquivo:
        df = carregar_arquivo(arquivo)

        if df is None:
            return

        st.subheader("📌 Prévia dos dados")
        st.dataframe(df.head())

        # selecionar colunas numéricas
        colunas_numericas = df.select_dtypes(include=np.number).columns

        if len(colunas_numericas) < 2:
            st.warning("O arquivo precisa ter pelo menos duas colunas numéricas.")
            return

        col_x = st.selectbox("Selecione coluna X", colunas_numericas)
        col_y = st.selectbox("Selecione coluna Y", colunas_numericas)

        if st.button("Gerar análise"):

            x = df[col_x].dropna()
            y = df[col_y].dropna()

            # métricas
            st.subheader("📈 Estatísticas")

            col1, col2, col3 = st.columns(3)

            col1.metric("Média X", f"{x.mean():.2f}")
            col2.metric("Mín X", f"{x.min():.2f}")
            col3.metric("Máx X", f"{x.max():.2f}")

            # gráficos
            st.subheader("📊 Gráficos")

            g1, g2, g3 = st.columns(3)

            # dispersão
            fig1, ax1 = plt.subplots()
            ax1.scatter(x, y)
            ax1.set_title("Dispersão")
            ax1.set_xlabel(col_x)
            ax1.set_ylabel(col_y)
            g1.pyplot(fig1)

            # histograma X
            fig2, ax2 = plt.subplots()
            ax2.hist(x, bins=20)
            ax2.set_title(f"Histograma - {col_x}")
            g2.pyplot(fig2)

            # histograma Y
            fig3, ax3 = plt.subplots()
            ax3.hist(y, bins=20)
            ax3.set_title(f"Histograma - {col_y}")
            g3.pyplot(fig3)

            # regressão linear
            st.subheader("📉 Regressão Linear")

            try:
                coef = np.polyfit(x, y, 1)
                poly1d_fn = np.poly1d(coef)

                fig4, ax4 = plt.subplots()
                ax4.scatter(x, y)
                ax4.plot(x, poly1d_fn(x))
                ax4.set_title("Regressão Linear")
                st.pyplot(fig4)

                st.success(f"Equação da reta: y = {coef[0]:.4f}x + {coef[1]:.4f}")

            except Exception as e:
                st.error(f"Erro ao calcular regressão: {e}")


if __name__ == "__main__":
    main()