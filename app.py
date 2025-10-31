import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Análise de COVID-19 no Brasil", layout="wide", initial_sidebar_state="expanded")

st.title("Análise de COVID-19 no Brasil")
st.markdown("""
Este aplicativo demonstra o uso de **Ciência de Dados aplicada à Saúde**, com dados reais da **COVID-19**.
""")

# =====================
# FUNÇÃO DE CARREGAMENTO — LEVE
# =====================
@st.cache_data
def carregar_dados():
    try:
        url = "https://raw.githubusercontent.com/caiocarneloz/covid-brasil-estados/main/covid_estados_2021.csv"
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["confirmed", "deaths"])
    return df

# =====================
# CARREGAMENTO
# =====================
dados = carregar_dados()

if dados.empty:
    st.error("Erro ao carregar dados. Tente novamente mais tarde.")
    st.stop()

# =====================
# DADOS ATUAIS (última data)
# =====================
dados_atuais = dados.sort_values("date").groupby("state", as_index=False).last()

# =====================
# MÉTRICAS GERAIS
# =====================
st.header("Dashboard Geral — Situação Atual")
total_casos = int(dados_atuais["confirmed"].sum())
total_mortes = int(dados_atuais["deaths"].sum())
letalidade = (total_mortes / total_casos * 100) if total_casos > 0 else 0

c1, c2, c3 = st.columns(3)
c1.metric("Casos Totais", f"{total_casos:,}".replace(",", "."))
c2.metric("Mortes Totais", f"{total_mortes:,}".replace(",", "."))
c3.metric("Letalidade (%)", f"{letalidade:.2f}")

# =====================
# DISTRIBUIÇÃO POR ESTADO
# =====================
st.subheader("Distribuição por Estado")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    ax1.bar(dados_atuais["state"], dados_atuais["confirmed"], color="#1f77b4")
    ax1.set_title("Casos Confirmados")
    plt.xticks(rotation=90)
    st.pyplot(fig1)
    plt.close(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.bar(dados_atuais["state"], dados_atuais["deaths"], color="#d62728")
    ax2.set_title("Mortes")
    plt.xticks(rotation=90)
    st.pyplot(fig2)
    plt.close(fig2)

# =====================
# FILTRO DE ESTADO
# =====================
st.sidebar.header("Filtros")
estados = sorted(dados["state"].unique())
estado_sel = st.sidebar.selectbox("Estado", estados)

# =====================
# EVOLUÇÃO TEMPORAL
# =====================
st.header(f"Evolução Temporal — {estado_sel}")
df_plot = dados[dados["state"] == estado_sel].sort_values("date")

if df_plot.empty:
    st.warning("Sem dados para exibir.")
else:
    serie = df_plot.groupby("date")[["confirmed", "deaths"]].sum().reset_index()

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(serie["date"], serie["confirmed"], label="Casos", color="#1f77b4")
    ax3.plot(serie["date"], serie["deaths"], label="Mortes", color="#d62728")
    ax3.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig3)
    plt.close(fig3)

    final = serie.iloc[-1]
    st.markdown(f"Casos: **{int(final['confirmed']):,}** • Mortes: **{int(final['deaths']):,}**".replace(",", "."))

# =====================
# AGRUPAMENTO K-MEANS
# =====================
st.header("Agrupamento (K-Means)")

agr = dados_atuais[["state", "confirmed", "deaths"]].dropna()

if len(agr) > 1:
    X = StandardScaler().fit_transform(agr[["confirmed", "deaths"]])
    k = min(3, len(agr))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    agr["Cluster"] = km.fit_predict(X)
    st.dataframe(agr.sort_values("Cluster"))

    fig4, ax4 = plt.subplots()
    for c in sorted(agr["Cluster"].unique()):
        grupo = agr[agr["Cluster"] == c]
        ax4.scatter(grupo["confirmed"], grupo["deaths"], label=f"Grupo {c}")
    ax4.set_xlabel("Casos Confirmados")
    ax4.set_ylabel("Mortes")
    ax4.legend()
    st.pyplot(fig4)
    plt.close(fig4)
else:
    st.warning("Dados insuficientes para clusterização.")

st.markdown("---")
st.caption("Projeto Acadêmico — Ciência de Dados em Saúde | 2025")
