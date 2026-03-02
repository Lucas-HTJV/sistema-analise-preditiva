import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from io import BytesIO

# =============================
# CONFIG
# =============================
st.set_page_config(layout="wide")
st.title("📊 Sistema de Análise Preditiva")

# =============================
# CSS (CARDS)
# =============================
st.markdown("""
<style>
.card {
    background-color: #111;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(255,0,0,0.3);
    border-left: 5px solid red;
}
.card-title {
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 10px;
    color: red;
}
.card p {
    margin: 5px 0;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# SIDEBAR
# =============================
st.sidebar.header("⚙️ Configurações")
arquivo = st.sidebar.file_uploader(
    "📂 Envie o arquivo",
    type=["xlsx", "xls", "csv", "json"]
)

# =============================
# FUNÇÕES
# =============================
def carregar_arquivo(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".json"):
            return pd.read_json(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        return None


def limpar_dados(df, col_x, col_y):
    df = df[[col_x, col_y]].copy()

    df = df.replace("Not Specified", np.nan)

    df[col_x] = pd.to_numeric(df[col_x], errors="coerce")
    df[col_y] = pd.to_numeric(df[col_y], errors="coerce")

    df = df.dropna()
    df = df[df[col_x] != 0]

    return df


def calcular_estatisticas(col):
    return {
        "Min": float(np.min(col)),
        "Max": float(np.max(col)),
        "Média": float(np.mean(col)),
        "Mediana": float(np.median(col))
    }


def gerar_pearson(df, col_x, col_y):
    x = df[col_x]
    y = df[col_y]

    mx = x.mean()
    my = y.mean()

    num = ((x - mx) * (y - my)).sum()
    dx = ((x - mx) ** 2).sum()
    dy = ((y - my) ** 2).sum()

    r = num / np.sqrt(dx * dy)

    return mx, my, num, dx, dy, r


# =============================
# PROCESSAMENTO
# =============================
if arquivo:

    df = carregar_arquivo(arquivo)

    if df is None or df.empty:
        st.warning("Arquivo inválido ou vazio.")
        st.stop()

    # PREVIEW
    st.subheader("📋 Prévia dos dados")
    st.dataframe(df.head(), use_container_width=True)

    colunas = df.columns.tolist()

    col_x = st.sidebar.selectbox("Coluna X", colunas)
    col_y = st.sidebar.selectbox("Coluna Y", colunas)

    if col_x == col_y:
        st.warning("Escolha colunas diferentes.")
        st.stop()

    df = limpar_dados(df, col_x, col_y)

    if df.empty or len(df) < 2:
        st.warning("Dados insuficientes.")
        st.stop()

    st.success(f"{len(df)} registros válidos")

    # =============================
    # ESTATÍSTICAS (CARDS)
    # =============================
    st.subheader("📊 Estatísticas")

    stats_x = calcular_estatisticas(df[col_x])
    stats_y = calcular_estatisticas(df[col_y])

    df["k"] = df[col_y] / df[col_x]

    k_min = df["k"].min()
    k_max = df["k"].max()
    k_med = df["k"].median()

    pearson = df[col_x].corr(df[col_y])

    c1, c2, c3 = st.columns(3)

    with c1:
        txt = "".join([f"<p><b>{k}:</b> {v:.4f}</p>" for k, v in stats_x.items()])
        st.markdown(f"<div class='card'><div class='card-title'>X</div>{txt}</div>", unsafe_allow_html=True)

    with c2:
        txt = "".join([f"<p><b>{k}:</b> {v:.4f}</p>" for k, v in stats_y.items()])
        st.markdown(f"<div class='card'><div class='card-title'>Y</div>{txt}</div>", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>Métricas</div>
            <p><b>Pearson:</b> {pearson:.4f}</p>
            <p><b>k min:</b> {k_min:.4e}</p>
            <p><b>k max:</b> {k_max:.4e}</p>
            <p><b>k mediana:</b> {k_med:.4e}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # =============================
    # PEARSON DETALHADO
    # =============================
    st.subheader("📐 Equação de Pearson")

    mx, my, num, dx, dy, r = gerar_pearson(df, col_x, col_y)

    st.latex(r"""
    r = \frac{\sum (X - \bar{X})(Y - \bar{Y})}
    {\sqrt{\sum (X - \bar{X})^2 \cdot \sum (Y - \bar{Y})^2}}
    """)

    st.latex(
        rf"""
    r = \frac{{{num:.4e}}}
    {{\sqrt{{{dx:.4e} \cdot {dy:.4e}}}}}
    """
    )

    st.latex(rf"r = {r:.4f}")

    st.divider()

    # =============================
    # REGRESSÃO
    # =============================
    X = df[[col_x]].values
    y = df[col_y].values

    modelo = LinearRegression()
    modelo.fit(X, y)

    y_pred = modelo.predict(X)
    r2 = modelo.score(X, y)

    a = modelo.intercept_
    b = modelo.coef_[0]

    df_log = df[(df[col_x] > 0) & (df[col_y] > 0)]

    if len(df_log) >= 2:
        log_x = np.log10(df_log[col_x].values.reshape(-1, 1))
        log_y = np.log10(df_log[col_y].values)

        modelo_log = LinearRegression()
        modelo_log.fit(log_x, log_y)

        r2_log = modelo_log.score(log_x, log_y)

        alpha = modelo_log.intercept_
        beta = modelo_log.coef_[0]
    else:
        log_x = None

    # =============================
    # GRÁFICOS
    # =============================
    st.subheader("📊 Gráficos")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.scatter(df, x=col_x, y=col_y)
        fig1.add_trace(go.Scatter(x=df[col_x], y=y_pred, mode="lines"))

        fig1.update_layout(title=f"y = {a:.4f} + {b:.4f}x | R²={r2:.4f}")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        if log_x is not None:
            fig2 = px.scatter(x=df_log[col_x], y=df_log[col_y], log_x=True, log_y=True)

            x_sorted = np.sort(df_log[col_x])
            y_line = 10 ** (alpha + beta * np.log10(x_sorted))

            fig2.add_trace(go.Scatter(x=x_sorted, y=y_line, mode="lines"))

            fig2.update_layout(
                title=f"log10(y) = {alpha:.4f} + {beta:.4f}log10(x) | R²={r2_log:.4f}"
            )

            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Sem dados positivos suficientes para log-log")

    # HISTOGRAMA
    st.subheader("📉 Distribuição de k")
    fig4 = px.histogram(df, x="k", nbins=30)
    st.plotly_chart(fig4, use_container_width=True)

    # EXPORTAÇÃO
    st.subheader("💾 Exportar dados")

    buffer = BytesIO()
    df.to_excel(buffer, index=False)

    st.download_button(
        "📥 Baixar Excel Completo",
        data=buffer.getvalue(),
        file_name="dados_tratados.xlsx"
    )
