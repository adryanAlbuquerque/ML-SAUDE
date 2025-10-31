import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------
# Título e descrição do app
# -------------------------
st.title("🩺 Análise de COVID-19 no Brasil")
st.markdown("""
Este aplicativo demonstra o uso de **Machine Learning aplicado à Saúde**, com foco na **COVID-19 no Brasil**.

Aqui você pode:
- Explorar os dados oficiais por estado e região;
- Visualizar gráficos de casos e mortes;
- Realizar uma análise de agrupamento (não supervisionada);
- Treinar um modelo simples de regressão (supervisionada) para prever casos.
""")

# -------------------------
# Carregar dados
# -------------------------
@st.cache_data
def carregar_dados():
    url = "https://raw.githubusercontent.com/caiocarneloz/brasilio-covid19-data/main/covid19_brasil_io.csv"
    df = pd.read_csv(url)
    return df

dados = carregar_dados()

st.subheader("📊 Prévia dos dados")
st.dataframe(dados.head())

# -------------------------
# Seleção de filtros
# -------------------------
estados = sorted(dados["state"].unique())
estado_selecionado = st.selectbox("Selecione um estado:", estados)

dados_estado = dados[dados["state"] == estado_selecionado]

# -------------------------
# Gráfico de evolução
# -------------------------
st.subheader(f"📈 Evolução de casos e mortes em {estado_selecionado}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dados_estado["date"], dados_estado["confirmed"], label="Casos Confirmados")
ax.plot(dados_estado["date"], dados_estado["deaths"], label="Mortes", color="red")
ax.set_xlabel("Data")
ax.set_ylabel("Quantidade")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# -------------------------
# Análise Não Supervisionada (K-Means)
# -------------------------
st.subheader("🧬 Agrupamento (K-Means)")
st.markdown("Agrupa estados com base no número total de casos e mortes.")

agrupamento = dados.groupby("state")[["confirmed", "deaths"]].max().reset_index()

scaler = StandardScaler()
dados_padronizados = scaler.fit_transform(agrupamento[["confirmed", "deaths"]])

modelo_kmeans = KMeans(n_clusters=3, random_state=42)
agrupamento["Cluster"] = modelo_kmeans.fit_predict(dados_padronizados)

st.dataframe(agrupamento)

fig2, ax2 = plt.subplots()
for cluster in sorted(agrupamento["Cluster"].unique()):
    grupo = agrupamento[agrupamento["Cluster"] == cluster]
    ax2.scatter(grupo["confirmed"], grupo["deaths"], label=f"Grupo {cluster}")
ax2.set_xlabel("Casos Confirmados")
ax2.set_ylabel("Mortes")
ax2.legend()
st.pyplot(fig2)

# -------------------------
# Análise Supervisionada (Regressão Linear)
# -------------------------
st.subheader("📉 Predição Simples de Casos")
st.markdown("Treina um modelo de regressão linear para prever o número de casos com base nas mortes registradas.")

X = dados[["deaths"]]
y = dados["confirmed"]

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_treino, y_treino)
pred = modelo.predict(X_teste)

r2 = r2_score(y_teste, pred)
mae = mean_absolute_error(y_teste, pred)

st.write(f"**R² (qualidade do ajuste):** {r2:.4f}")
st.write(f"**Erro Médio Absoluto (MAE):** {mae:.2f}")

st.markdown("#### Gráfico de previsão")
fig3, ax3 = plt.subplots()
ax3.scatter(y_teste, pred, alpha=0.5)
ax3.set_xlabel("Casos reais")
ax3.set_ylabel("Casos previstos")
ax3.set_title("Previsão de Casos de COVID-19")
st.pyplot(fig3)
