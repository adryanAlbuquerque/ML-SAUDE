import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------
# Configuração da página
# -------------------------
st.set_page_config(
    page_title="Análise de COVID-19 no Brasil",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Estilo / cabeçalho
# -------------------------
st.title("Análise de COVID-19 no Brasil")
st.markdown("""
Este painel interativo demonstra o uso de **Ciência de Dados aplicada à Saúde**, utilizando dados reais da **COVID-19 no Brasil**.

**Objetivos:**
- Coletar e tratar dados reais da pandemia;
- Explorar indicadores nacionais, estaduais e municipais;
- Aplicar técnicas de análise exploratória e modelagem não supervisionada;
- Comunicar resultados de forma clara e acessível.
""")

# -------------------------
# Carregar e tratar dados
# -------------------------
@st.cache_data
def carregar_dados():
    url = "https://data.brasil.io/dataset/covid19/caso.csv.gz"
    df = pd.read_csv(url, compression="gzip", low_memory=False)
    # Tratamento básico
    df = df.dropna(subset=["confirmed", "deaths"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["is_last"] == True]  # mantém apenas último registro por local
    return df

dados = carregar_dados()

# -------------------------
# Resumo geral dos dados
# -------------------------
st.header("Visão Geral dos Dados")

col1, col2, col3 = st.columns(3)
total_casos = int(dados["confirmed"].sum())
total_mortes = int(dados["deaths"].sum())
letalidade = (total_mortes / total_casos) * 100 if total_casos > 0 else 0

col1.metric("Casos Totais", f"{total_casos:,}".replace(",", "."))
col2.metric("Mortes Totais", f"{total_mortes:,}".replace(",", "."))
col3.metric("Letalidade Média (%)", f"{letalidade:.2f}")

# -------------------------
# Dashboard geral por estado
# -------------------------
st.subheader("Panorama Nacional por Estado")

dados_estados = dados[dados["place_type"] == "state"].copy()
dados_estados = dados_estados.groupby("state")[["confirmed", "deaths"]].sum().reset_index()
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
# Filtros laterais
# -------------------------
st.sidebar.header("Filtros de Análise")

nivel = st.sidebar.radio("Nível de Análise", ["Estado", "Cidade"])
if nivel == "Estado":
    estados = sorted(dados.loc[dados["place_type"] == "state", "state"].unique())
    estado_sel = st.sidebar.selectbox("Selecione um Estado", estados)
    dados_estado = dados[(dados["place_type"] == "state") & (dados["state"] == estado_sel)]
else:
    estados = sorted(dados.loc[dados["place_type"] == "city", "state"].unique())
    estado_sel = st.sidebar.selectbox("Selecione um Estado", estados)
    cidades = sorted(dados.loc[(dados["place_type"] == "city") & (dados["state"] == estado_sel), "city"].dropna().unique())
    cidade_sel = st.sidebar.selectbox("Selecione uma Cidade", cidades)
    dados_estado = dados[(dados["place_type"] == "city") & (dados["state"] == estado_sel) & (dados["city"] == cidade_sel)]

# -------------------------
# Evolução temporal
# -------------------------
st.header(f"Evolução Temporal — {estado_sel}" + (f" / {cidade_sel}" if nivel == "Cidade" else ""))

url_full = "https://data.brasil.io/dataset/covid19/caso_full.csv.gz"
@st.cache_data
def carregar_dados_full():
    df = pd.read_csv(url_full, compression="gzip", low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    return df

dados_full = carregar_dados_full()

if nivel == "Estado":
    df_plot = dados_full[(dados_full["place_type"] == "state") & (dados_full["state"] == estado_sel)]
else:
    df_plot = dados_full[(dados_full["place_type"] == "city") &
                         (dados_full["state"] == estado_sel) &
                         (dados_full["city"] == cidade_sel)]

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(df_plot["date"], df_plot["confirmed"], label="Casos Confirmados", color="tab:blue")
ax3.plot(df_plot["date"], df_plot["deaths"], label="Mortes", color="tab:red")
ax3.set_title("Evolução de Casos e Mortes")
ax3.set_xlabel("Data")
ax3.set_ylabel("Quantidade")
ax3.legend()
plt.xticks(rotation=45)
st.pyplot(fig3)

# -------------------------
# Agrupamento (K-Means)
# -------------------------
st.header("Análise de Agrupamento (K-Means)")
st.markdown("""
Agrupa os estados com base em **características epidemiológicas**:
- Total de casos confirmados
- Total de mortes registradas
""")

dados_kmeans = dados_estados[["state", "confirmed", "deaths"]].copy()
scaler = StandardScaler()
dados_padronizados = scaler.fit_transform(dados_kmeans[["confirmed", "deaths"]])

modelo = KMeans(n_clusters=3, random_state=42)
dados_kmeans["Cluster"] = modelo.fit_predict(dados_padronizados)

fig4, ax4 = plt.subplots()
for cluster in sorted(dados_kmeans["Cluster"].unique()):
    grupo = dados_kmeans[dados_kmeans["Cluster"] == cluster]
    ax4.scatter(grupo["confirmed"], grupo["deaths"], label=f"Grupo {cluster}")
ax4.set_xlabel("Casos Confirmados")
ax4.set_ylabel("Mortes")
ax4.set_title("Agrupamento de Estados (K-Means)")
ax4.legend()
st.pyplot(fig4)

st.markdown("""
**Interpretação:**  
- Estados no mesmo grupo compartilham perfis semelhantes de impacto da pandemia.  
- Grupos com valores mais altos representam regiões mais afetadas.  
""")

# -------------------------
# Rodapé
# -------------------------
st.markdown("---")
st.caption("Projeto de Análise de Dados — Disciplina de Ciência de Dados | 2025")
