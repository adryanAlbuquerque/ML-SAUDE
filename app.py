import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------
# Configuração da página
# -------------------------
st.set_page_config(
    page_title="Análise de COVID-19 no Brasil",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Cabeçalho
# -------------------------
st.title("Análise de COVID-19 no Brasil")
st.markdown(
    """
    Este aplicativo demonstra o uso de **Machine Learning aplicado à Saúde**, com foco na **COVID-19 no Brasil**.

    Funcionalidades:
    - Explorar os dados oficiais por estado ou município;
    - Visualizar gráficos de casos e mortes;
    - Analisar agrupamentos (K-Means);
    - Treinar um modelo de regressão linear para prever casos.
    """
)

# -------------------------
# Carregar dados
# -------------------------
@st.cache_data
def carregar_dados():
    url = "https://data.brasil.io/dataset/covid19/caso.csv.gz"
    df = pd.read_csv(url, compression='gzip', low_memory=False)
    return df

dados = carregar_dados()

# -------------------------
# Barra lateral — filtros
# -------------------------
st.sidebar.header("Filtros")

# Seleção de nível
place_type = st.sidebar.radio(
    "Nível de análise",
    options=["state", "city"],
    format_func=lambda x: "Estado" if x == "state" else "Município"
)

# Seleção de estado
estados_disponiveis = sorted(dados[dados["place_type"] == "state"]["state"].unique())
estado_selecionado = st.sidebar.selectbox("Selecione o estado:", estados_disponiveis)

# Filtro base
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

# -------------------------
# Visualização da amostra
# -------------------------
st.subheader("Pré-visualização dos dados filtrados")
st.dataframe(dados_filtrados.head())

# -------------------------
# Gráfico de evolução temporal
# -------------------------
titulo_local = estado_selecionado if not cidade_selecionada else f"{cidade_selecionada} / {estado_selecionado}"
st.subheader(f"Evolução de casos e mortes em {titulo_local}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(pd.to_datetime(dados_filtrados["date"]), dados_filtrados["confirmed"], label="Casos Confirmados")
ax.plot(pd.to_datetime(dados_filtrados["date"]), dados_filtrados["deaths"], label="Mortes", color="red")
ax.set_xlabel("Data")
ax.set_ylabel("Quantidade")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# -------------------------
# Análise de Agrupamento (K-Means)
# -------------------------
if place_type == "state":
    st.subheader("Análise de Agrupamento (K-Means)")
    st.markdown("Agrupa estados com base no número máximo de casos e mortes registrados.")

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

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for cluster in sorted(agrupamento["Cluster"].unique()):
        grupo = agrupamento[agrupamento["Cluster"] == cluster]
        ax2.scatter(grupo["confirmed"], grupo["deaths"], label=f"Grupo {cluster}")
    ax2.set_xlabel("Casos Confirmados (máx)")
    ax2.set_ylabel("Mortes (máx)")
    ax2.legend()
    st.pyplot(fig2)
else:
    st.info("A análise de agrupamento está disponível apenas para o nível de estado.")

# -------------------------
# Regressão Linear — previsão
# -------------------------
if place_type == "state":
    st.subheader("Predição de Casos com Regressão Linear")
    st.markdown("Treina um modelo para prever o número de casos com base nas mortes registradas.")

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
    col1.metric("R² (Qualidade do ajuste)", f"{r2:.4f}")
    col2.metric("Erro Médio Absoluto (MAE)", f"{mae:,.0f}")

    st.markdown("#### Gráfico de previsão")
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.scatter(y_test, pred, alpha=0.6)
    ax3.set_xlabel("Casos Reais")
    ax3.set_ylabel("Casos Previstos")
    ax3.set_title("Previsão de Casos de COVID-19 (Regressão Linear)")
    st.pyplot(fig3)
else:
    st.info("A predição está disponível apenas no nível de estado.")

# -------------------------
# Rodapé
# -------------------------
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 - Projeto Educacional de Análise de Dados em Saúde")
