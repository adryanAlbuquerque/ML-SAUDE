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

@st.cache_data
def carregar_dados():
    try:
        url = "https://data.brasil.io/dataset/covid19/caso.csv.gz"
        df = pd.read_csv(url, compression="gzip", low_memory=False)
    except:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cols = ["city", "state", "place_type", "date", "confirmed", "deaths", "is_last"]
    df = df[[c for c in cols if c in df.columns]]
    df["confirmed"] = pd.to_numeric(df["confirmed"], errors="coerce")
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")
    if "place_type" in df.columns:
        df = df[df["place_type"].isin(["state", "city"])]
    df = df.dropna(subset=["confirmed", "deaths"])
    return df

dados = carregar_dados()

if dados.empty:
    st.error("Erro ao carregar dados do Brasil.IO. Tente novamente mais tarde.")
    st.stop()

if "is_last" in dados.columns:
    dados_atuais = dados[dados["is_last"] == True].copy()
    if dados_atuais.empty:
        dados_atuais = dados.sort_values("date").groupby(["state", "place_type", "city"], as_index=False).last()
else:
    dados_atuais = dados.sort_values("date").groupby(["state", "place_type", "city"], as_index=False).last()

st.header("Dashboard Geral — Situação Atual")
total_casos = int(dados_atuais["confirmed"].sum())
total_mortes = int(dados_atuais["deaths"].sum())
letalidade = (total_mortes / total_casos * 100) if total_casos > 0 else 0

c1, c2, c3 = st.columns(3)
c1.metric("Casos Totais", f"{total_casos:,}".replace(",", "."))
c2.metric("Mortes Totais", f"{total_mortes:,}".replace(",", "."))
c3.metric("Letalidade (%)", f"{letalidade:.2f}")

st.subheader("Distribuição por Estado")
dados_estados = dados_atuais[dados_atuais["place_type"] == "state"].copy()
dados_estados["letalidade"] = (dados_estados["deaths"] / dados_estados["confirmed"]).replace([float("inf"), -float("inf")], 0) * 100

col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    ax1.bar(dados_estados["state"], dados_estados["confirmed"])
    ax1.set_title("Casos por Estado")
    plt.xticks(rotation=90)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.bar(dados_estados["state"], dados_estados["deaths"])
    ax2.set_title("Mortes por Estado")
    plt.xticks(rotation=90)
    st.pyplot(fig2)

st.sidebar.header("Filtros")
nivel = st.sidebar.radio("Nível", ["Estado", "Cidade"])

if nivel == "Estado":
    estados = sorted(dados[dados["place_type"] == "state"]["state"].unique())
    estado_sel = st.sidebar.selectbox("Estado", estados)
    cidade_sel = None
    df_plot = dados[(dados["place_type"] == "state") & (dados["state"] == estado_sel)]
else:
    estados = sorted(dados[dados["place_type"] == "city"]["state"].unique())
    estado_sel = st.sidebar.selectbox("Estado", estados)
    cidades = sorted(dados[(dados["place_type"] == "city") & (dados["state"] == estado_sel)]["city"].unique())
    cidade_sel = st.sidebar.selectbox("Cidade", cidades)
    df_plot = dados[(dados["place_type"] == "city") & (dados["state"] == estado_sel) & (dados["city"] == cidade_sel)]

st.header(f"Evolução — {estado_sel}" + (f" / {cidade_sel}" if cidade_sel else ""))

df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")
df_plot = df_plot.dropna(subset=["date"]).sort_values("date")

if df_plot.empty:
    st.warning("Sem dados")
else:
    serie = df_plot.groupby("date")[["confirmed", "deaths"]].sum().reset_index()
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(serie["date"], serie["confirmed"], label="Casos")
    ax3.plot(serie["date"], serie["deaths"], label="Mortes")
    plt.xticks(rotation=45)
    ax3.legend()
    st.pyplot(fig3)
    final = serie.iloc[-1]
    st.markdown(f"Casos: **{int(final['confirmed']):,}** • Mortes: **{int(final['deaths']):,}**".replace(",", "."))

st.header("Agrupamento (K-Means)")

agr = dados_atuais[dados_atuais["place_type"] == "state"][["state", "confirmed", "deaths"]].dropna()
if len(agr) > 1:
    scaler = StandardScaler()
    X = scaler.fit_transform(agr[["confirmed", "deaths"]])
    k = min(3, len(agr))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    agr["Cluster"] = km.fit_predict(X)
    st.dataframe(agr.sort_values("Cluster"))
    fig4, ax4 = plt.subplots()
    for c in sorted(agr["Cluster"].unique()):
        g = agr[agr["Cluster"] == c]
        ax4.scatter(g["confirmed"], g["deaths"], label=f"Grupo {c}")
    ax4.legend()
    st.pyplot(fig4)
else:
    st.warning("Dados insuficientes")

st.markdown("---")
st.caption("Projeto Acadêmico — Ciência de Dados em Saúde | 2025")
