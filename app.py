import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------
# CONFIGURAÇÃO DA PÁGINA
# -------------------------
st.set_page_config(
    page_title="Análise de COVID-19 no Brasil",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# CABEÇALHO
# -------------------------
st.title("Análise de COVID-19 no Brasil")
st.markdown("""
Este aplicativo demonstra o uso de **Ciência de Dados aplicada à Saúde**, com dados reais da **COVID-19**.

**Você poderá:**
- Explorar os dados por **estado** ou **município**;
- Visualizar **gráficos de casos e mortes**;
- Ver um **painel geral com indicadores nacionais**;
- Aplicar **agrupamento (K-Means)** para análise exploratória;
- Interpretar e comunicar resultados de forma clara.
""")

# -------------------------
# CARREGAR DADOS
# -------------------------
@st.cache_data
def carregar_dados():
    url = "https://data.brasil.io/dataset/covid19/caso.csv.gz"
    df = pd.read_csv(url, compression="gzip", low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["confirmed", "deaths"])
    return df

dados = carregar_dados()

# -------------------------
# TRATAMENTO DE DADOS
# -------------------------
dados = dados[dados["place_type"].isin(["state", "city"])]

# Último registro de cada local (situação atual)
dados_atuais = dados.sort_values("date").groupby(["state", "place_type", "city"], as_index=False).last()

# -------------------------
# DASHBOARD GERAL
# -------------------------
st.header("Dashboard Geral — Situação Atual")

# Agregado nacional
total_casos = int(dados_atuais["confirmed"].sum())
total_mortes = int(dados_atuais["deaths"].sum())
letalidade = (total_mortes / total_casos * 100) if total_casos > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Casos Totais", f"{total_casos:,}".replace(",", "."))
col2.metric("Mortes Totais", f"{total_mortes:,}".replace(",", "."))
col3.metric("Letalidade Média (%)", f"{letalidade:.2f}")

# Panorama por estado
st.subheader("Distribuição por Estado")

dados_estados = dados_atuais[dados_atuais["place_type"] == "state"].copy()
dados_estados["letalidade"] = (dados_estados["deaths"] / dados_estados["confirmed"]) * 100

col_g1, col_g2 = st.columns(2)

with col_g1:
    fig1, ax1 = plt.subplots()
    ax1.bar(dados_estados["state"], dados_estados["confirmed"], color="tab:blue")
    ax1.set_title("Casos Confirmados por Estado")
    ax1.set_xlabel("Estado")
    ax1.set_ylabel("Casos")
    plt.xticks(rotation=90)
    st.pyplot(fig1)

with col_g2:
    fig2, ax2 = plt.subplots()
    ax2.bar(dados_estados["state"], dados_estados["deaths"], color="tab:red")
    ax2.set_title("Mortes por Estado")
    ax2.set_xlabel("Estado")
    ax2.set_ylabel("Mortes")
    plt.xticks(rotation=90)
    st.pyplot(fig2)

# -------------------------
# FILTROS DE ANÁLISE
# -------------------------
st.sidebar.header("Filtros de Análise")

nivel = st.sidebar.radio("Nível de Análise", ["Estado", "Cidade"])

if nivel == "Estado":
    estados = sorted(dados[dados["place_type"] == "state"]["state"].unique())
    estado_sel = st.sidebar.selectbox("Selecione um Estado", estados)
    cidade_sel = None
    dados_filtrados = dados[(dados["place_type"] == "state") & (dados["state"] == estado_sel)]
else:
    estados = sorted(dados[dados["place_type"] == "city"]["state"].unique())
    estado_sel = st.sidebar.selectbox("Selecione um Estado", estados)
    cidades = sorted(
        dados[(dados["place_type"] == "city") & (dados["state"] == estado_sel)]["city"].dropna().unique()
    )
    cidade_sel = st.sidebar.selectbox("Selecione uma Cidade", cidades)
    dados_filtrados = dados[
        (dados["place_type"] == "city")
        & (dados["state"] == estado_sel)
        & (dados["city"] == cidade_sel)
    ]

# -------------------------
# EVOLUÇÃO TEMPORAL
# -------------------------
st.header(f"Evolução Temporal — {estado_sel}" + (f" / {cidade_sel}" if cidade_sel else ""))

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(pd.to_datetime(dados_filtrados["date"]), dados_filtrados["confirmed"], label="Casos Confirmados", color="tab:blue")
ax3.plot(pd.to_datetime(dados_filtrados["date"]), dados_filtrados["deaths"], label="Mortes", color="tab:red")
ax3.set_xlabel("Data")
ax3.set_ylabel("Quantidade")
ax3.legend()
ax3.set_title("Evolução de Casos e Mortes")
plt.xticks(rotation=45)
st.pyplot(fig3)

# -------------------------
# ANÁLISE NÃO SUPERVISIONADA — K-MEANS
# -------------------------
st.header("Análise de Agrupamento (K-Means)")
st.markdown("""
Agrupamento dos **estados brasileiros** com base em:
- Casos confirmados acumulados
- Mortes registradas

Essa técnica ajuda a **identificar perfis semelhantes** de impacto da pandemia.
""")

agrupamento = dados_atuais[dados_atuais["place_type"] == "state"][["state", "confirmed", "deaths"]].copy()

scaler = StandardScaler()
dados_padronizados = scaler.fit_transform(agrupamento[["confirmed", "deaths"]])

modelo_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
agrupamento["Cluster"] = modelo_kmeans.fit_predict(dados_padronizados)

st.dataframe(agrupamento)

fig4, ax4 = plt.subplots()
for cluster in sorted(agrupamento["Cluster"].unique()):
    grupo = agrupamento[agrupamento["Cluster"] == cluster]
    ax4.scatter(grupo["confirmed"], grupo["deaths"], label=f"Grupo {cluster}")
ax4.set_xlabel("Casos Confirmados")
ax4.set_ylabel("Mortes")
ax4.set_title("Agrupamento de Estados (K-Means)")
ax4.legend()
st.pyplot(fig4)

st.markdown("""
**Interpretação:**
- Estados no mesmo grupo compartilham padrões semelhantes de casos e mortes.
- Grupos com maiores valores indicam maior impacto da COVID-19.
""")

# -------------------------
# RODAPÉ
# -------------------------
st.markdown("---")
st.caption("Projeto Acadêmico — Ciência de Dados em Saúde | 2025")
