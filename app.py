# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# =====================
# CONFIGURAÇÃO GERAL
# =====================
st.set_page_config(page_title="Análise COVID-19 no Brasil", layout="wide")
sns.set_style("whitegrid")

st.title("📊 Análise Interativa da COVID-19 no Brasil")
st.markdown("""
Aplicação de Ciência de Dados em Saúde — dados de [wcota/covid19br](https://github.com/wcota/covid19br).
""")

# =====================
# CARREGAMENTO DE DADOS
# =====================
@st.cache_data
def carregar_dados():
    url = "https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv"
    df = pd.read_csv(url)
    df = df.rename(columns={"state": "estado", "totalCases": "casos", "deaths": "obitos", "date": "data"})
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.dropna(subset=["estado", "casos", "obitos"])
    estados_validos = [
        "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
        "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
        "RO", "RR", "RS", "SC", "SE", "SP", "TO"
    ]
    df = df[df["estado"].isin(estados_validos)]
    return df

dados = carregar_dados()
dados_atuais = dados.sort_values("data").groupby("estado", as_index=False).last()
dados_atuais["letalidade"] = (dados_atuais["obitos"] / dados_atuais["casos"]).fillna(0) * 100

# =====================
# SIDEBAR
# =====================
st.sidebar.title("📂 Navegação")
pagina = st.sidebar.radio(
    "Selecione uma seção:",
    ["Visão Geral", "Tendências por Estado", "Agrupamento (K-Means)", "Modelagem Supervisionada"]
)

# =====================
# VISÃO GERAL
# =====================
if pagina == "Visão Geral":
    st.header("Panorama Geral da COVID-19 no Brasil")

    total_casos = int(dados_atuais["casos"].sum())
    total_obitos = int(dados_atuais["obitos"].sum())
    letalidade_geral = (total_obitos / total_casos) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Casos Totais", f"{total_casos:,}".replace(",", "."))
    c2.metric("Óbitos Totais", f"{total_obitos:,}".replace(",", "."))
    c3.metric("Letalidade Média (%)", f"{letalidade_geral:.2f}")

    st.markdown("### Estados com Maior Número de Casos")
    top_casos = dados_atuais.nlargest(10, "casos")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(y="estado", x="casos", data=top_casos, palette="Blues_r", ax=ax)
    ax.set_xlabel("Casos Confirmados")
    ax.set_ylabel("Estado")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Taxa de Letalidade por Estado (%)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="estado", y="letalidade", data=dados_atuais.sort_values("letalidade", ascending=False), palette="Reds_r", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)
    plt.close(fig)

    st.info("🔍 **Interpretação:** estados com letalidade mais alta podem indicar atraso na detecção de casos leves ou sobrecarga no sistema de saúde.")

# =====================
# TENDÊNCIAS POR ESTADO
# =====================
elif pagina == "Tendências por Estado":
    st.header("Evolução Temporal por Estado")

    estado_sel = st.selectbox("Selecione o Estado", sorted(dados["estado"].unique()), index=25)
    df_estado = dados[dados["estado"] == estado_sel].sort_values("data")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(df_estado["data"], df_estado["casos"], alpha=0.4, color="skyblue", label="Casos")
    ax.plot(df_estado["data"], df_estado["casos"], color="blue", linewidth=2)
    ax.set_title(f"Evolução dos Casos — {estado_sel}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Casos Confirmados")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Óbitos Acumulados")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(df_estado["data"], df_estado["obitos"], alpha=0.4, color="salmon", label="Óbitos")
    ax.plot(df_estado["data"], df_estado["obitos"], color="red", linewidth=2)
    ax.set_xlabel("Data")
    ax.set_ylabel("Óbitos")
    st.pyplot(fig)
    plt.close(fig)

    st.success("📈 Gráficos de área facilitam a visualização de crescimento acumulado e ajudam a identificar picos ou desacelerações.")

# =====================
# AGRUPAMENTO K-MEANS
# =====================
elif pagina == "Agrupamento (K-Means)":
    st.header("Agrupamento de Estados por Perfil de Casos e Óbitos")

    X = dados_atuais[["casos", "obitos"]]
    Xs = StandardScaler().fit_transform(X)
    k = st.slider("Número de grupos (k)", 2, 6, 3)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(Xs)
    dados_atuais["Cluster"] = clusters

    st.markdown("### Distribuição dos Estados por Cluster")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x="casos", y="obitos", hue="Cluster", data=dados_atuais, palette="Set2", s=100, ax=ax
    )
    for i, row in dados_atuais.iterrows():
        ax.text(row["casos"], row["obitos"], row["estado"], fontsize=8)
    ax.set_title("Clusters de Estados segundo Casos e Óbitos")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Média de Indicadores por Cluster")
    st.dataframe(
        dados_atuais.groupby("Cluster")[["casos", "obitos", "letalidade"]].mean().round(2)
    )

    st.info("💡 Estados com comportamento semelhante (em escala) ficam no mesmo grupo, facilitando comparações regionais.")

# =====================
# MODELAGEM SUPERVISIONADA
# =====================
elif pagina == "Modelagem Supervisionada":
    st.header("Predição de Óbitos a partir de Casos Confirmados")

    st.markdown("Nesta análise simples, usamos **Regressão Linear** para estimar o número de óbitos com base na quantidade de casos confirmados por estado.")

    X = dados_atuais[["casos"]].values
    y = dados_atuais["obitos"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2:.2f}")
    c2.metric("MAE", f"{mae:.1f}")
    c3.metric("RMSE", f"{rmse:.1f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.regplot(x=y_test, y=y_pred, line_kws={"color": "gray"}, scatter_kws={"s": 80})
    ax.set_xlabel("Óbitos Reais")
    ax.set_ylabel("Óbitos Preditos")
    ax.set_title("Desempenho da Regressão Linear")
    st.pyplot(fig)
    plt.close(fig)

    st.success("✅ Relação linear forte indica que o número de casos é um bom preditor do total de óbitos (tendência esperada).")

# =====================
# RODAPÉ
# =====================
st.markdown("---")
st.caption("Projeto Acadêmico — Ciência de Dados em Saúde | 2025")
