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
    page_title="🩺 Análise de COVID-19 no Brasil",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Estilo / cabeçalho
# -------------------------
st.title("🩺 Análise de COVID-19 no Brasil")
st.markdown(
    """
    Este aplicativo demonstra o uso de **Machine Learning aplicado à Saúde**, com foco na **COVID-19 no Brasil**.<br>
    Aqui você pode:
    - Explorar os dados oficiais por estado (e opcionalmente por município);
    - Visualizar gráficos de casos e mortes;
    - Realizar uma análise de agrupamento (K-Means);
    - Treinar um modelo simples de regressão para prever casos.
    """,
    unsafe_allow_html=True
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

# Mostra preview
st.subheader("📊 Pré-visualização dos dados")
st.dataframe(dados.head())

# -------------------------
# Filtros de seleção
# -------------------------
st.sidebar.header("Filtrar dados")

# Níveis: estado ou município
place_type = st.sidebar.selectbox(
    "Nível de agregação",
    options=["state", "city"],
    help="Escolha se deseja analisar por estado ('state') ou por município ('city')"
)

if place_type == "state":
    op_states = sorted(dados.loc[dados["place_type"] == "state", "state"].unique())
    estado_selecionado = st.sidebar.selectbox("Selecione um estado:", op_states)
    dados_filtrados = dados[(dados["place_type"] == "state") & (dados["state"] == estado_selecionado)]
    cidade_selecionada = None
else:
    # nível município: primeiro escolher estado, depois município
    op_states = sorted(dados.loc[dados["place_type"] == "city", "state"].unique())
    estado_selecionado = st.sidebar.selectbox("Selecione um estado:", op_states)
    op_cities = sorted(dados.loc[(dados["place_type"] == "city") & (dados["state"] == estado_selecionado), "city"].dropna().unique())
    cidade_selecionada = st.sidebar.selectbox("Selecione uma cidade:", op_cities)
    dados_filtrados = dados[(dados["place_type"] == "city") &
                             (dados["state"] == estado_selecionado) &
                             (dados["city"] == cidade_selecionada)]

# -------------------------
# Gráfico de evolução
# -------------------------
titulo_local = f"{estado_selecionado}" + (f" / {cidade_selecionada}" if cidade_selecionada else "")
st.subheader(f"📈 Evolução de casos e mortes em {titulo_local}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(pd.to_datetime(dados_filtrados["date"]), dados_filtrados["confirmed"], label="Casos Confirmados")
ax.plot(pd.to_datetime(dados_filtrados["date"]), dados_filtrados["deaths"], label="Mortes", color="red")
ax.set_xlabel("Data")
ax.set_ylabel("Quantidade")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# -------------------------
# Análise Não Supervisionada (K-Means) — somente por estado faz sentido
# -------------------------
if place_type == "state":
    st.subheader("🧬 Agrupamento (K-Means)")
    st.markdown("Agrupa estados com base no número máximo de casos e mortes registrados.")

    agrupamento = (dados[dados["place_type"] == "state"]
                   .groupby("state")[["confirmed", "deaths"]]
                   .max()
                   .reset_index())

    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(agrupamento[["confirmed", "deaths"]])

    modelo_kmeans = KMeans(n_clusters=3, random_state=42)
    agrupamento["Cluster"] = modelo_kmeans.fit_predict(dados_padronizados)

    st.dataframe(agrupamento)

    fig2, ax2 = plt.subplots()
    for cluster in sorted(agrupamento["Cluster"].unique()):
        grupo = agrupamento[agrupamento["Cluster"] == cluster]
        ax2.scatter(grupo["confirmed"], grupo["deaths"], label=f"Grupo {cluster}")
    ax2.set_xlabel("Casos Confirmados Máx")
    ax2.set_ylabel("Mortes Máx")
    ax2.legend()
    st.pyplot(fig2)
else:
    st.info("A análise de agrupamento está disponível apenas no nível **estado**.")

# -------------------------
# Análise Supervisionada (Regressão Linear) — também somente por estado
# -------------------------
if place_type == "state":
    st.subheader("📉 Predição Simples de Casos")
    st.markdown("Treina um modelo de regressão linear para prever o número de casos com base nas mortes registradas.")

    df_model = dados[(dados["place_type"] == "state")][["deaths", "confirmed"]].dropna()
    X = df_model[["deaths"]]
    y = df_model["confirmed"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    st.write(f"**R² (qualidade do ajuste):** {r2:.4f}")
    st.write(f"**Erro Médio Absoluto (MAE):** {mae:.2f}")

    st.markdown("#### Gráfico de previsão")
    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test, pred, alpha=0.5)
    ax3.set_xlabel("Casos reais")
    ax3.set_ylabel("Casos previstos")
    ax3.set_title("Previsão de Casos de COVID-19")
    st.pyplot(fig3)
else:
    st.info("A predição está disponível apenas no nível **estado**.")

# -------------------------
# Rodapé
# -------------------------
st.sidebar.markdown("---")
st.sidebar.write("© 2025 Seu Nome / Projeto de Estudo")

