# 📊 Sistema de Análise Preditiva

Sistema desenvolvido em Python utilizando Streamlit para análise estatística e modelagem preditiva com regressão linear e regressão em escala logarítmica (log–log).

---

## 🎯 Objetivo

Fornecer uma ferramenta interativa para:

- Análise exploratória de dados
- Cálculo de estatísticas descritivas
- Cálculo do coeficiente de correlação de Pearson
- Regressão Linear
- Regressão Log-Log
- Visualização gráfica interativa
- Exportação de resultados em Excel

---

## 🚀 Tecnologias Utilizadas

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- ReportLab
- XlsxWriter

---

## 📂 Estrutura do Projeto
Projeto/
│
├── app_streamlit.py
├── requirements.txt
├── README.md
└── Data/

## 📊 Funcionalidades

### ✔ Upload de arquivos
Suporte para:
- `.xlsx`
- `.xls`
- `.csv`
- `.json`

### ✔ Análise Estatística
- Mínimo
- Máximo
- Média
- Mediana
- Coeficiente k
- Correlação de Pearson

### ✔ Modelagem Preditiva
- Regressão Linear (y = a + bx)
- Regressão Log-Log (log10(y) = α + β log10(x))
- Cálculo de R²

### ✔ Visualizações
- Gráfico de dispersão com linha de regressão
- Gráfico em escala log-log
- Histograma da distribuição de k

### ✔ Exportações
- Excel com dados tratados e resultados
- Relatório PDF automático

---

## ⚙️ Como Executar o Projeto

### 1️⃣ Clonar o repositório

cd Projeto
python main.py
