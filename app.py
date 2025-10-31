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
    # Mantém apenas colunas que usamos, se existirem
    cols_want = ["city", "state", "place_type", "date", "confirmed", "deaths", "is_last"]
    cols_present = [c for c in cols_want if c in df.columns]
    df = df[cols_present]
    # Converte para numérico (garante que confirmed/deaths sejam numéricos)
    if "confirmed" in df.columns:
        df["confirmed"] = pd.to_numeric(df["confirmed"], errors="coerce")
    if "deaths" in df.columns:
        df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")
    # Filtra apenas state/city quando houver essa coluna
    if "place_type" in df.columns:
        df = df[df["place_type"].isin(["state", "city"])]
    # Remove linhas sem números
    df = df.dropna(subset=[c for c in ["confirmed", "deaths"] if c in df.columns])
    return df

dados = carregar_dados()

# Verifica carregamento
if dados.empty:
    st.error("Não foi possível carregar dados válidos do Brasil.IO. Tente novamente mais tarde.")
    st.stop()

# -------------------------
# TRATAMENTO DE DADOS
# -------------------------
# Usamos o último registro por local (estado/cidade) para o dashboard atual.
# Se 'is_last' existir, podemos usá-lo; senão usamos groupby+last por date.
if "is_last" in dados.columns:
    dados_atuais = dados[dados["is_last"] == True].copy()
    # como alguns locais podem não ter is_last marcado, garantir last por agrupamento
    if dados_atuais.empty:
        dados_atuais = dados.sort_values("date").groupby(["state", "place_type", "city"], as_index=False).last()
else:
    dados_atuais = dados.sort_values("date").groupby(["state", "place_type", "city"], as_index=False).last()

# -------------------------
# DASHBOARD GERAL
# -------------------------
st.header("Dashboard Geral — Situação Atual")

total_casos = int(dados_atuais["confirmed"].sum()) if "confirmed" in dados_atuais.columns else 0
total_mortes = int(dados_atuais["deaths"].sum()) if "deaths" in dados_atuais.columns else 0
letalidade = (total_mortes / total_casos * 100) if total_casos > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Casos Totais", f"{total_casos:,}".replace(",", "."))
col2.metric("Mortes Totais", f"{total_mortes:,}".replace(",", "."))
col3.metric("Letalidade Média (%)", f"{letalidade:.2f}")

# -------------------------
# PANORAMA POR ESTADO
# -------------------------
st.subheader("Distribuição por Estado")

dados_estados = dados_atuais[dados_atuais["place_type"] == "state"].copy()
# caso as colunas existam, calcular letalidade por estado
if {"confirmed", "deaths"}.issubset(dados_estados.columns):
    dados_estados["letalidade"] = (dados_estados["deaths"] / dados_estados["confirmed"]).replace([float("inf"), -float("inf")], 0) * 100
else:
    dados_estados["letalidade"] = 0

col_g1, col_g2 = st.columns(2)
with col_g1:
    fig1, ax1 = plt.subplots()
    if not dados_estados.empty and "confirmed" in dados_estados.columns:
        ax1.bar(dados_estados["state"], dados_estados["confirmed"], color="tab:blue")
        ax1.set_title("Casos Confirmados por Estado")
        ax1.set_xlabel("Estado")
        ax1.set_ylabel("Casos")
        plt.xticks(rotation=90)
        st.pyplot(fig1)
    else:
        st.info("Não há dados de casos por estado para exibir o gráfico.")

with col_g2:
    fig2, ax2 = plt.subplots()
    if not dados_estados.empty and "deaths" in dados_estados.columns:
        ax2.bar(dados_estados["state"], dados_estados["deaths"], color="tab:red")
        ax2.set_title("Mortes por Estado")
        ax2.set_xlabel("Estado")
        ax2.set_ylabel("Mortes")
        plt.xticks(rotation=90)
        st.pyplot(fig2)
    else:
        st.info("Não há dados de mortes por estado para exibir o gráfico.")

# -------------------------
# FILTROS LATERAIS
# -------------------------
st.sidebar.header("Filtros de Análise")

nivel = st.sidebar.radio("Nível de Análise", ["Estado", "Cidade"])
if nivel == "Estado":
    estados = sorted(dados[dados["place_type"] == "state"]["state"].dropna().unique())
    if not estados:
        st.error("Não há estados disponíveis nos dados.")
        st.stop()
    estado_sel = st.sidebar.selectbox("Selecione um Estado", estados)
    cidade_sel = None
    dados_filtrados = dados[(dados["place_type"] == "state") & (dados["state"] == estado_sel)]
else:
    estados = sorted(dados[dados["place_type"] == "city"]["state"].dropna().unique())
    if not estados:
        st.error("Não há cidades disponíveis nos dados.")
        st.stop()
    estado_sel = st.sidebar.selectbox("Selecione um Estado", estados)
    cidades = sorted(dados[(dados["place_type"] == "city") & (dados["state"] == estado_sel)]["city"].dropna().unique())
    if not cidades:
        st.error("Não há municípios para o estado selecionado.")
        st.stop()
    cidade_sel = st.sidebar.selectbox("Selecione uma Cidade", cidades)
    dados_filtrados = dados[
        (dados["place_type"] == "city") &
        (dados["state"] == estado_sel) &
        (dados["city"] == cidade_sel)
    ]

# -------------------------
# EVOLUÇÃO TEMPORAL (SUBSTITUÍDA POR UMA ANÁLISE FUNCIONAL)
# -------------------------
st.header(f"Evolução Temporal — {estado_sel}" + (f" / {cidade_sel}" if cidade_sel else ""))

# Filtra dados de acordo com o nível
if nivel == "Estado":
    df_plot = dados[(dados["place_type"] == "state") & (dados["state"] == estado_sel)].copy()
else:
    df_plot = dados[
        (dados["place_type"] == "city") &
        (dados["state"] == estado_sel) &
        (dados["city"] == cidade_sel)
    ].copy()

# Garante formato de data e ordenação
df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")
df_plot = df_plot.dropna(subset=["date"]).sort_values("date")

# Verifica se há dados
if df_plot.empty or "confirmed" not in df_plot.columns or "deaths" not in df_plot.columns:
    st.warning("Sem dados disponíveis para gerar a evolução temporal.")
else:
    # Agrupa por data (em caso de duplicações)
    serie = df_plot.groupby("date")[["confirmed", "deaths"]].sum().reset_index()

    # Gráfico de linhas
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(serie["date"], serie["confirmed"], label="Casos Confirmados", color="tab:blue")
    ax3.plot(serie["date"], serie["deaths"], label="Mortes", color="tab:red")
    ax3.set_title("Evolução de Casos e Mortes")
    ax3.set_xlabel("Data")
    ax3.set_ylabel("Quantidade")
    ax3.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # Estatísticas adicionais
    total_final = serie.iloc[-1]
    st.markdown(f"""
    **Resumo:**
    - Período analisado: {serie["date"].min().date()} — {serie["date"].max().date()}
    - Casos acumulados: **{int(total_final["confirmed"]):,}**
    - Mortes acumuladas: **{int(total_final["deaths"]):,}**
    """.replace(",", "."))

# -------------------------
# ANÁLISE DE AGRUPAMENTO (K-MEANS) - com checagens
# -------------------------
st.header("Análise de Agrupamento (K-Means)")
st.markdown("""
Agrupa os estados com base em:
- Casos confirmados acumulados
- Mortes registradas
""")

# Prepara DataFrame para kmeans
agrupamento = dados_atuais[dados_atuais["place_type"] == "state"][["state", "confirmed", "deaths"]].copy()

# Sanity checks antes de padronizar/clusterizar
if agrupamento.empty:
    st.warning("Não há dados de estados para aplicar K-Means.")
else:
    # garante tipo numérico e remove linhas com NaN em confirmed/deaths
    agrupamento["confirmed"] = pd.to_numeric(agrupamento["confirmed"], errors="coerce")
    agrupamento["deaths"] = pd.to_numeric(agrupamento["deaths"], errors="coerce")
    agrupamento = agrupamento.dropna(subset=["confirmed", "deaths"])

    n_samples = len(agrupamento)
    if n_samples < 2:
        st.warning("Número insuficiente de estados com dados numéricos (precisa de pelo menos 2) para K-Means.")
    else:
        # define n_clusters seguro
        n_clusters = min(3, n_samples)
        scaler = StandardScaler()
        dados_padronizados = scaler.fit_transform(agrupamento[["confirmed", "deaths"]])

        # executar kmeans com n_init explícito (compatibilidade sklearn)
        modelo_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        agrupamento["Cluster"] = modelo_kmeans.fit_predict(dados_padronizados)

        st.dataframe(agrupamento.sort_values("Cluster"))

        fig4, ax4 = plt.subplots(figsize=(8, 5))
        for cluster in sorted(agrupamento["Cluster"].unique()):
            grupo = agrupamento[agrupamento["Cluster"] == cluster]
            ax4.scatter(grupo["confirmed"], grupo["deaths"], label=f"Grupo {cluster}")
        ax4.set_xlabel("Casos Confirmados")
        ax4.set_ylabel("Mortes")
        ax4.set_title(f"Agrupamento de Estados (K-Means, k={n_clusters})")
        ax4.legend()
        st.pyplot(fig4)

        st.markdown("""
        **Interpretação:**
        - Estados no mesmo grupo compartilham perfis semelhantes de impacto.
        - Grupos com maiores valores representam maior impacto da COVID-19.
        """)

# -------------------------
# RODAPÉ
# -------------------------
st.markdown("---")
st.caption("Projeto Acadêmico — Ciência de Dados em Saúde | 2025")