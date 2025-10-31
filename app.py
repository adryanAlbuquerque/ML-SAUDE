import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# =============================
# Configuração da Página
# =============================
st.set_page_config(
    page_title="Análise de COVID-19 no Brasil",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Análise de COVID-19 no Brasil")
st.markdown("""
Aplicativo interativo para explorar dados da COVID-19 no Brasil.  
Inclui dashboards, gráficos, análises de agrupamento e predição com Machine Learning.
""")

# =============================
# Carregar Dados
# =============================
@st.cache_data
def carregar_dados():
    url = "https://data.brasil.io/dataset/covid19/caso.csv.gz"
    df = pd.read_csv(url, compression='gzip', low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    return df

dados = carregar_dados()

# =============================
# Aba principal (Tabs)
# =============================
aba = st.tabs([
    "Dashboard Geral",
    "Análise Detalhada",
    "Agrupamento (K-Means)",
    "Predição (Regressão Linear)"
])

# ==========================================================
# DASHBOARD GERAL
# ==========================================================
with aba[0]:
    st.header("Dashboard Geral")

    # Somente nível de estado
    dados_estados = dados[dados["place_type"] == "state"].copy()

    # Última data por estado
    ultimos = dados_estados[dados_estados["is_last"] == True].copy()

    # Indicadores nacionais
    total_confirmed = ultimos["confirmed"].sum()
    total_deaths = ultimos["deaths"].sum()
    fatality_rate = (total_deaths / total_confirmed) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Casos Confirmados (Brasil)", f"{total_confirmed:,.0f}")
    col2.metric("Total de Mortes", f"{total_deaths:,.0f}")
    col3.metric("Letalidade Média", f"{fatality_rate:.2f}%")

    # Gráfico de casos por estado
    st.subheader("Casos Confirmados por Estado")
    casos_estado = ultimos.sort_values("confirmed", ascending=False)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(casos_estado["state"], casos_estado["confirmed"], color="steelblue")
    ax1.set_xlabel("Estado")
    ax1.set_ylabel("Casos Confirmados")
    ax1.set_title("Total de Casos Confirmados por Estado")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Gráfico de mortes por estado
    st.subheader("Mortes por Estado")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(casos_estado["state"], casos_estado["deaths"], color="firebrick")
    ax2.set_xlabel("Estado")
    ax2.set_ylabel("Mortes")
    ax2.set_title("Total de Mortes por Estado")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Ranking de estados com mais casos e mortes
    st.subheader("Ranking dos Estados (Top 10)")
    top_casos = casos_estado.nlargest(10, "confirmed")[["state", "confirmed", "deaths"]]
    st.dataframe(top_casos)

# ==========================================================
# ANÁLISE DETALHADA
# ==========================================================
with aba[1]:
    st.header("Análise Detalhada por Estado ou Município")

    st.sidebar.header("Filtros de Análise Detalhada")

    place_type = st.sidebar.radio(
        "Nível de análise",
        options=["state", "city"],
        format_func=lambda x: "Estado" if x == "state" else "Município"
    )

    estados_disponiveis = sorted(dados[dados["place_type"] == "state"]["state"].unique())
    estado_selecionado = st.sidebar.selectbox("Selecione o estado:", estados_disponiveis)

    if place_type == "state":
        dados_filtrados = dados[(dados["place_type"] == "state") & (dados["state"] == estado_selecionado)]
        cidade_selecionada = None
    else:
        cidades_disponiveis = sorted(
            dados[(dados["place_type"] == "city") & (dados["state"] == estado_selecionado)]["city"].dropna().unique()
        )
        cidade_selecionada = st.sidebar.selectbox("Selecione o município:", cidades_disponiveis)
        dados_filtrados = dados[
            (dados["place_type"] == "city")
            & (dados["state"] == estado_selecionado)
            & (dados["city"] == cidade_selecionada)
        ]

    titulo_local = estado_selecionado if not cidade_selecionada else f"{cidade_selecionada} / {estado_selecionado}"

    st.subheader(f"Evolução de Casos e Mortes em {titulo_local}")
    st.dataframe(dados_filtrados.head())

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(dados_filtrados["date"], dados_filtrados["confirmed"], label="Casos Confirmados")
    ax3.plot(dados_filtrados["date"], dados_filtrados["deaths"], label="Mortes", color="red")
    ax3.set_xlabel("Data")
    ax3.set_ylabel("Quantidade")
    ax3.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig3)

# ==========================================================
# AGRUPAMENTO (K-MEANS)
# ==========================================================
with aba[2]:
    st.header("Análise de Agrupamento (K-Means)")

    agrupamento = (
        dados[dados["place_type"] == "state"]
        .groupby("state")[["confirmed", "deaths"]]
        .max()
        .reset_index()
    )

    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(agrupamento[["confirmed", "deaths"]])

    modelo_kmeans = KMeans(n_clusters=3, random_state=42)
    agrupamento["Cluster"] = modelo_kmeans.fit_predict(dados_padronizados)

    st.dataframe(agrupamento)

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    for cluster in sorted(agrupamento["Cluster"].unique()):
        grupo = agrupamento[agrupamento["Cluster"] == cluster]
        ax4.scatter(grupo["confirmed"], grupo["deaths"], label=f"Grupo {cluster}")
    ax4.set_xlabel("Casos Confirmados (máx)")
    ax4.set_ylabel("Mortes (máx)")
    ax4.legend()
    st.pyplot(fig4)

# ==========================================================
# REGRESSÃO LINEAR
# ==========================================================
with aba[3]:
    st.header("Predição de Casos com Regressão Linear")

    df_model = dados[(dados["place_type"] == "state")][["deaths", "confirmed"]].dropna()
    X = df_model[["deaths"]]
    y = df_model["confirmed"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    col1, col2 = st.columns(2)
    col1.metric("R² (Qualidade do Ajuste)", f"{r2:.4f}")
    col2.metric("Erro Médio Absoluto (MAE)", f"{mae:,.0f}")

    st.subheader("Gráfico de Previsão")
    fig5, ax5 = plt.subplots(figsize=(7, 5))
    ax5.scatter(y_test, pred, alpha=0.6)
    ax5.set_xlabel("Casos Reais")
    ax5.set_ylabel("Casos Previstos")
    ax5.set_title("Previsão de Casos de COVID-19 (Regressão Linear)")
    st.pyplot(fig5)

# ==========================================================
# Rodapé
# ==========================================================
st.markdown("---")
st.caption("© 2025 - Projeto Educacional de Análise de Dados em Saúde")
