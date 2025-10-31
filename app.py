import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Análise de COVID-19 no Brasil", layout="wide")

st.title("Análise de COVID-19 no Brasil")

@st.cache_data
def carregar_dados():
    url = "https://data.brasil.io/dataset/covid19/caso.csv.gz"
    df = pd.read_csv(
        url,
        compression="gzip",
        usecols=["state", "place_type", "city", "date", "confirmed", "deaths", "is_last"]
    )
    df = df[df["place_type"] == "state"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["confirmed"] = pd.to_numeric(df["confirmed"], errors="coerce")
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")
    df = df.dropna(subset=["confirmed", "deaths"])
    return df

dados = carregar_dados()

if dados.empty:
    st.error("Erro ao carregar dados.")
    st.stop()

if "is_last" in dados.columns:
    dados_atuais = dados[dados["is_last"] == True].copy()
    if dados_atuais.empty:
        dados_atuais = dados.sort_values("date").groupby(["state"], as_index=False).last()
else:
    dados_atuais = dados.sort_values("date").groupby(["state"], as_index=False).last()

total_casos = int(dados_atuais["confirmed"].sum())
total_mortes = int(dados_atuais["deaths"].sum())
letalidade = (total_mortes / total_casos) * 100 if total_casos > 0 else 0

c1, c2, c3 = st.columns(3)
c1.metric("Casos Totais", f"{total_casos:,}".replace(",", "."))
c2.metric("Mortes Totais", f"{total_mortes:,}".replace(",", "."))
c3.metric("Letalidade (%)", f"{letalidade:.2f}")

st.subheader("Casos e Mortes por Estado")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    ax.bar(dados_atuais["state"], dados_atuais["confirmed"])
    plt.xticks(rotation=90)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.bar(dados_atuais["state"], dados_atuais["deaths"])
    plt.xticks(rotation=90)
    st.pyplot(fig)

st.header("Evolução Temporal por Estado")
estados = sorted(dados["state"].unique())
estado_sel = st.selectbox("Estado", estados)

df_plot = dados[dados["state"] == estado_sel].copy()
df_plot = df_plot.sort_values("date")
serie = df_plot.groupby("date")[["confirmed", "deaths"]].sum().reset_index()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(serie["date"], serie["confirmed"], label="Casos")
ax.plot(serie["date"], serie["deaths"], label="Mortes")
plt.xticks(rotation=45)
ax.legend()
st.pyplot(fig)

final = serie.iloc[-1]
st.markdown(f"Casos: **{int(final['confirmed']):,}** • Mortes: **{int(final['deaths']):,}**".replace(",", "."))

st.header("Agrupamento (K-Means)")

agr = dados_atuais[["state", "confirmed", "deaths"]].dropna()
if len(agr) > 1:
    X = StandardScaler().fit_transform(agr[["confirmed", "deaths"]])
    k = min(3, len(agr))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    agr["Cluster"] = km.fit_predict(X)
    st.dataframe(agr.sort_values("Cluster"))

    fig, ax = plt.subplots()
    for c in sorted(agr["Cluster"].unique()):
        g = agr[agr["Cluster"] == c]
        ax.scatter(g["confirmed"], g["deaths"], label=f"Grupo {c}")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Dados insuficientes para clusterização.")

st.markdown("---")
st.caption("Projeto Acadêmico — Ciência de Dados em Saúde | 2025")
