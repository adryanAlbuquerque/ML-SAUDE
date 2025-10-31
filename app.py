import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Configuração inicial
st.set_page_config(page_title="Análise de COVID-19 no Brasil", layout="wide")

st.title("Análise de COVID-19 no Brasil")
st.markdown("""
Este projeto demonstra a aplicação de **Aprendizado de Máquina Supervisionado** em dados reais da **COVID-19 no Brasil**.  
O objetivo é analisar, visualizar e prever o número de óbitos com base na quantidade de casos confirmados.
""")

# Carregar dados
@st.cache_data
def carregar_dados():
    url = "https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv"
    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"state": "estado", "totalCases": "casos", "deaths": "obitos"})
    df = df[["date", "estado", "casos", "obitos"]]
    df = df[df["estado"] != "TOTAL"]
    return df

dados = carregar_dados()

# Menu lateral
pagina = st.sidebar.radio("Navegação", ["Exploração dos Dados", "Análise por Estado", "Modelagem Supervisionada"])

# =======================
# EXPLORAÇÃO DOS DADOS
# =======================
if pagina == "Exploração dos Dados":
    st.header("Exploração dos Dados")

    dados_atuais = dados.groupby("estado")[["casos", "obitos"]].max().reset_index()
    dados_atuais["letalidade"] = (dados_atuais["obitos"] / dados_atuais["casos"]) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Casos Totais", f"{dados_atuais['casos'].sum():,}".replace(",", "."))
    c2.metric("Óbitos Totais", f"{dados_atuais['obitos'].sum():,}".replace(",", "."))
    c3.metric("Letalidade Média (%)", f"{dados_atuais['letalidade'].mean():.2f}")

    st.markdown("### Casos confirmados por estado")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dados_atuais.sort_values("casos", ascending=False), x="estado", y="casos", color="steelblue")
    ax.set_xlabel("Estado")
    ax.set_ylabel("Casos Confirmados")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Interpretação:** Os estados com maior número de casos são São Paulo, Minas Gerais e Paraná, devido à alta densidade populacional.")

    st.markdown("### Óbitos confirmados por estado")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dados_atuais.sort_values("obitos", ascending=False), x="estado", y="obitos", color="indianred")
    ax.set_xlabel("Estado")
    ax.set_ylabel("Óbitos")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Interpretação:** O padrão de óbitos acompanha os casos, com destaque para os estados mais populosos.")

    st.markdown("### Taxa de letalidade (%) por estado")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dados_atuais.sort_values("letalidade", ascending=False), x="estado", y="letalidade", color="gray")
    ax.set_xlabel("Estado")
    ax.set_ylabel("Letalidade (%)")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Interpretação:** Estados com letalidade mais alta podem ter menor capacidade de testagem ou atendimento médico.")

# =======================
# ANÁLISE INDIVIDUAL
# =======================
elif pagina == "Análise por Estado":
    st.header("Análise Individual por Estado")
    estados = sorted(dados["estado"].unique())
    estado_sel = st.selectbox("Selecione o estado:", estados)
    df_estado = dados[dados["estado"] == estado_sel]

    st.markdown(f"### Evolução da COVID-19 em {estado_sel}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_estado["date"], df_estado["casos"], label="Casos", color="steelblue")
    ax.plot(df_estado["date"], df_estado["obitos"], label="Óbitos", color="indianred")
    ax.legend()
    ax.set_xlabel("Data")
    ax.set_ylabel("Quantidade")
    st.pyplot(fig)
    plt.close(fig)

    taxa_crescimento = df_estado["casos"].pct_change().mean() * 100
    st.metric("Crescimento médio diário de casos (%)", f"{taxa_crescimento:.2f}")

    st.markdown("**Interpretação:** Este gráfico mostra o crescimento acumulado de casos e óbitos ao longo do tempo. A taxa de crescimento indica o avanço médio diário da doença no estado selecionado.")

# =======================
# MODELAGEM SUPERVISIONADA
# =======================
elif pagina == "Modelagem Supervisionada":
    st.header("Predição de Óbitos com Regressão Linear")

    st.markdown("""
    Nesta etapa, aplicamos um modelo de **Regressão Linear** para prever o número de óbitos com base no total de casos confirmados.
    """)

    dados_atuais = dados.groupby("estado")[["casos", "obitos"]].max().reset_index()
    X = dados_atuais[["casos"]]
    y = dados_atuais["obitos"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{r2:.2f}")
    col2.metric("MAE", f"{mae:.0f}")
    col3.metric("RMSE", f"{rmse:.0f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(x=y_test, y=y_pred, scatter_kws={"s": 80, "color": "steelblue"}, line_kws={"color": "red"})
    ax.set_xlabel("Óbitos Reais")
    ax.set_ylabel("Óbitos Preditos")
    ax.set_title("Desempenho da Regressão Linear")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    **Interpretação:**  
    - O **R²** indica quanto da variação dos óbitos é explicada pelos casos confirmados.  
    - O **MAE** e o **RMSE** mostram o erro médio das previsões.  
    - Como o modelo usa apenas uma variável (casos), ele captura a relação geral, mas não considera fatores externos como testagem, idade média ou vacinação.
    """)

# Rodapé
st.markdown("---")
st.caption("Projeto Acadêmico — Machine Learning Aplicado à Saúde | 2025")
