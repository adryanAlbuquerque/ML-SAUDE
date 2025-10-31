import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =====================
# CONFIGURAÇÕES GERAIS
# =====================
st.set_page_config(page_title="Análise de COVID-19 no Brasil", layout="wide", initial_sidebar_state="expanded")

st.title("🦠 Análise de COVID-19 no Brasil")
st.markdown("""
Aplicativo interativo de **Ciência de Dados aplicada à Saúde**, com foco na **COVID-19** no Brasil.  
Os dados são obtidos de uma fonte leve e confiável: [wcota/covid19br](https://github.com/wcota/covid19br).
""")

# =====================
# FUNÇÃO DE CARREGAMENTO DE DADOS
# =====================
@st.cache_data
def carregar_dados():
    try:
        url = "https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv"
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

    df = df.rename(columns={
        "date": "date",
        "state": "state",
        "totalCases": "confirmed",
        "deaths": "deaths"
    })

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["confirmed"] = pd.to_numeric(df["confirmed"], errors="coerce")
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")
    df = df.dropna(subset=["confirmed", "deaths"])

    estados_validos = [
        "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
        "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
        "RO", "RR", "RS", "SC", "SE", "SP", "TO"
    ]
    df = df[df["state"].isin(estados_validos)]
    return df

dados = carregar_dados()
if dados.empty:
    st.error("Erro ao carregar dados. Verifique sua conexão com a internet.")
    st.stop()

dados_atuais = dados.sort_values("date").groupby("state", as_index=False).last()

# =====================
# SEÇÃO 1: ANÁLISE GERAL DO BRASIL
# =====================
st.header("📊 Análise Geral do Brasil")

total_casos = int(dados_atuais["confirmed"].sum())
total_mortes = int(dados_atuais["deaths"].sum())
letalidade = (total_mortes / total_casos * 100) if total_casos > 0 else 0

c1, c2, c3 = st.columns(3)
c1.metric("Casos Totais", f"{total_casos:,}".replace(",", "."), help="Número total de casos confirmados no Brasil.")
c2.metric("Mortes Totais", f"{total_mortes:,}".replace(",", "."), help="Número total de óbitos confirmados.")
c3.metric("Letalidade (%)", f"{letalidade:.2f}", help="Percentual de mortes entre os casos confirmados.")

st.markdown(f"🧾 **Interpretação:** O Brasil registrou **{total_casos:,}** casos e **{total_mortes:,}** mortes, resultando em uma taxa de letalidade de aproximadamente **{letalidade:.2f}%**.".replace(",", "."))

# =====================
# GRÁFICOS GERAIS
# =====================
st.subheader("📍 Distribuição por Estado")

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    ax.bar(dados_atuais["state"], dados_atuais["confirmed"], color="#1f77b4")
    ax.set_title("Casos Confirmados por Estado")
    plt.xticks(rotation=90)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots()
    ax.bar(dados_atuais["state"], dados_atuais["deaths"], color="#d62728")
    ax.set_title("Mortes por Estado")
    plt.xticks(rotation=90)
    st.pyplot(fig)
    plt.close(fig)

# Letalidade por estado
st.subheader("☠️ Taxa de Letalidade por Estado")
dados_atuais["letalidade"] = (dados_atuais["deaths"] / dados_atuais["confirmed"]) * 100
dados_atuais["letalidade"] = dados_atuais["letalidade"].fillna(0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(dados_atuais["state"], dados_atuais["letalidade"], color="#ff7f0e")
ax.set_ylabel("% Letalidade")
ax.set_title("Taxa de Letalidade por Estado (%)")
plt.xticks(rotation=90)
st.pyplot(fig)
plt.close(fig)

st.markdown("""
🧩 **Interpretação:**  
Estados com letalidade mais alta indicam **maior gravidade clínica** ou **subnotificação de casos leves**.  
Já estados com letalidade menor podem ter **melhor capacidade de testagem ou atendimento médico.**
""")

# =====================
# SEÇÃO 2: ANÁLISE INDIVIDUAL POR ESTADO
# =====================
st.markdown("---")
st.header("📈 Análise Individual por Estado")

estados = sorted(dados["state"].unique())
estado_sel = st.selectbox("Selecione o Estado", estados, index=estados.index("SP"))

df_estado = dados[dados["state"] == estado_sel].sort_values("date")

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_estado["date"], df_estado["confirmed"], color="#1f77b4", label="Casos")
    ax.set_title(f"Evolução de Casos — {estado_sel}")
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_estado["date"], df_estado["deaths"], color="#d62728", label="Mortes")
    ax.set_title(f"Evolução de Mortes — {estado_sel}")
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# Comparativo do crescimento percentual diário
df_estado["novos_casos"] = df_estado["confirmed"].diff().fillna(0)
df_estado["crescimento_%"] = df_estado["novos_casos"].pct_change() * 100
media_crescimento = df_estado["crescimento_%"].replace([float("inf"), -float("inf")], 0).mean()

st.metric("📈 Crescimento médio diário (%)", f"{media_crescimento:.2f}")

st.markdown(f"""
📊 **Interpretação:**  
O estado **{estado_sel}** apresentou um crescimento médio diário de **{media_crescimento:.2f}%** nos casos durante o período analisado.  
Gráficos mostram a evolução temporal e ajudam a identificar **tendências de estabilização ou novos surtos**.
""")

# =====================
# SEÇÃO 3: AGRUPAMENTO K-MEANS
# =====================
st.markdown("---")
st.header("🤖 Agrupamento de Estados (K-Means)")

agr = dados_atuais[["state", "confirmed", "deaths"]].dropna()

if len(agr) > 1:
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
    ax.set_title("Agrupamento de Estados por Casos e Mortes")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
🧠 **Interpretação:**  
O agrupamento (K-Means) separa os estados em **grupos com características semelhantes**:  
- **Grupo 0:** Estados com números baixos de casos e mortes.  
- **Grupo 1:** Estados intermediários.  
- **Grupo 2:** Estados com os maiores números absolutos (ex: SP, RJ).  
""")
else:
    st.warning("Dados insuficientes para realizar o agrupamento.")

# =====================
# RODAPÉ
# =====================
st.markdown("---")
st.caption("Projeto Acadêmico — Ciência de Dados em Saúde | 2025")
