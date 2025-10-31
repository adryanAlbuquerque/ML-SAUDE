import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =====================
# CONFIGURAÇÃO GERAL
# =====================
st.set_page_config(page_title="Análise de COVID-19 no Brasil", layout="wide")
st.title("📊 Análise de COVID-19 no Brasil (Versão Leve)")

# =====================
# CARREGAMENTO DE DADOS
# =====================
@st.cache_data
def carregar_dados():
    url = "https://raw.githubusercontent.com/caiocarneloz/covid-brasil-estados/main/covid_estados_2021.csv"
    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["confirmed", "deaths"])
    return df

dados = carregar_dados()

if dados.empty:
    st.error("Erro ao carregar dados.")
    st.stop()

# =====================
# DADOS ATUAIS (última data)
# =====================
dados_atuais = dados.sort_values("date").groupby("state", as_index=False).last()

total_casos = int(dados_atuais["confirmed"].sum())
total_mortes = int(dados_atuais["deaths"].sum())
letalidade = (total_mortes / total_casos) * 100 if total_casos > 0 else 0

# =====================
# MÉTRICAS GERAIS
# =====================
c1, c2, c3 = st.columns(3)
c1.metric("Casos Totais", f"{total_casos:,}".replace(",", "."))
c2.metric("Mortes Totais", f"{total_mortes:,}".replace(",", "."))
c3.metric("Letalidade (%)", f"{letalidade:.2f}")

# =====================
# GRÁFICOS POR ESTADO
# =====================
st.subheader("Casos e Mortes por Estado")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    ax.bar(dados_atuais["state"], dados_atuais["confirmed"], color="#1f77b4")
    ax.set_title("Casos Confirmados")
    plt.xticks(rotation=90)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots()
    ax.bar(dados_atuais["state"], dados_atuais["deaths"], color="#d62728")
    ax.set_title("Mortes")
    plt.xticks(rotation=90)
    st.pyplot(fig)
    plt.close(fig)

# =====================
# EVOLUÇÃO TEMPORAL
# =====================
st.header("📈 Evolução Temporal por Estado")

estados = sorted(dados["state"].unique())
estado_sel = st.selectbox("Selecione o Estado", estados)

df_plot = dados[dados["state"] == estado_sel].sort_values("date")
serie = df_plot.groupby("date")[["confirmed", "deaths"]].sum().reset_index()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(serie["date"], serie["confirmed"], label="Casos", color="#1f77b4")
ax.plot(serie["date"], serie["deaths"], label="Mortes", color="#d62728")
ax.set_xlabel("Data")
ax.set_ylabel("Número de Casos / Mortes")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)
plt.close(fig)

final = serie.iloc[-1]
st.markdown(f"Casos: **{int(final['confirmed']):,}** • Mortes: **{int(final['deaths']):,}**".replace(",", "."))

# =====================
# AGRUPAMENTO (K-MEANS)
# =====================
st.header("🔍 Agrupamento de Estados (K-Means)")

agr = dados_atuais[["state", "confirmed", "deaths"]].dropna()

if len(agr) > 2:
    X = StandardScaler().fit_transform(agr[["confirmed", "deaths"]])
    k = min(3, len(agr))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    agr["Cluster"] = km.fit_predict(X)

    st.dataframe(agr.sort_values("Cluster"))

    fig, ax = plt.subplots()
    for c in sorted(agr["Cluster"].unique()):
        grupo = agr[agr["Cluster"] == c]
        ax.scatter(grupo["confirmed"], grupo["deaths"], label=f"Grupo {c}")
    ax.set_xlabel("Casos Confirmados")
    ax.set_ylabel("Mortes")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("Dados insuficientes para realizar a clusterização.")

st.markdown("---")
st.caption("Projeto Acadêmico — Ciência de Dados em Saúde | 2025")
