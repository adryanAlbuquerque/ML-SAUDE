
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


st.set_page_config(
    page_title="Análise de COVID-19 no Brasil",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Análise de COVID-19 no Brasil")
st.markdown("""
Aplicativo interativo de Ciência de Dados aplicada à Saúde, com foco na COVID-19 no Brasil.  
Os dados são obtidos de uma fonte pública: [wcota/covid19br](https://github.com/wcota/covid19br).
""")


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
    df = df.dropna(subset=["date", "state", "confirmed", "deaths"])


    estados_validos = [
        "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
        "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
        "RO", "RR", "RS", "SC", "SE", "SP", "TO"
    ]
    df = df[df["state"].isin(estados_validos)].copy()

    return df

dados = carregar_dados()
if dados.empty:
    st.error("Erro ao carregar dados. Verifique sua conexão com a internet.")
    st.stop()


dados_atuais = dados.sort_values("date").groupby("state", as_index=False).last().copy()
dados_atuais["letalidade"] = (dados_atuais["deaths"] / dados_atuais["confirmed"]).fillna(0) * 100


st.sidebar.title("Navegação")
pagina = st.sidebar.radio(
    "Selecione a visualização",
    ("Visão Geral", "Análise por Estado", "Agrupamento (K-Means)", "Aprendizagem Supervisionada")
)

if pagina == "Visão Geral":
    st.header("Visão Geral do Brasil")

    total_casos = int(dados_atuais["confirmed"].sum())
    total_mortes = int(dados_atuais["deaths"].sum())
    letalidade = (total_mortes / total_casos * 100) if total_casos > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Casos Totais", f"{total_casos:,}".replace(",", "."))
    c2.metric("Mortes Totais", f"{total_mortes:,}".replace(",", "."))
    c3.metric("Taxa de Letalidade (%)", f"{letalidade:.2f}")

    st.markdown(f"Resumo: {total_casos:,} casos e {total_mortes:,} mortes no Brasil (último registro disponível).".replace(",", "."))


    st.subheader("Distribuição por Estado")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(dados_atuais["state"], dados_atuais["confirmed"])
        ax.set_title("Casos Confirmados por Estado")
        ax.set_xlabel("Estado")
        ax.set_ylabel("Casos Confirmados")
        plt.xticks(rotation=90)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(dados_atuais["state"], dados_atuais["deaths"])
        ax.set_title("Mortes por Estado")
        ax.set_xlabel("Estado")
        ax.set_ylabel("Óbitos")
        plt.xticks(rotation=90)
        st.pyplot(fig)
        plt.close(fig)


    st.subheader("Taxa de Letalidade por Estado (%)")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(dados_atuais["state"], dados_atuais["letalidade"])
    ax.set_ylabel("% Letalidade")
    ax.set_title("Letalidade por Estado")
    plt.xticks(rotation=90)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        "Interpretação: Estados com maior letalidade podem indicar equipes de saúde saturadas, "
        "subnotificação de casos leves, ou atraso nas notificações. Estados com baixa letalidade "
        "podem ter melhor sistema de testagem/atenção."
    )


elif pagina == "Análise por Estado":
    st.header("Análise Individual por Estado")

    estados = sorted(dados["state"].unique())
    default_idx = 0
    if "SP" in estados:
        default_idx = estados.index("SP")
    estado_sel = st.selectbox("Selecione o Estado", estados, index=default_idx)

    df_estado = dados[dados["state"] == estado_sel].sort_values("date").copy()

    st.subheader(f"Evolução Temporal — {estado_sel}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_estado["date"], df_estado["confirmed"], label="Casos", linewidth=2)
        ax.set_title(f"Evolução de Casos — {estado_sel}")
        ax.set_xlabel("Data")
        ax.set_ylabel("Casos Confirmados")
        plt.xticks(rotation=45)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_estado["date"], df_estado["deaths"], label="Mortes", linewidth=2)
        ax.set_title(f"Evolução de Mortes — {estado_sel}")
        ax.set_xlabel("Data")
        ax.set_ylabel("Óbitos")
        plt.xticks(rotation=45)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)


    df_estado["novos_casos"] = df_estado["confirmed"].diff().fillna(0)

    df_estado["crescimento_%"] = df_estado["novos_casos"].pct_change().replace([np.inf, -np.inf], np.nan) * 100
    media_crescimento = df_estado["crescimento_%"].replace([np.nan], 0).mean()

    st.metric("Crescimento médio diário (%)", f"{media_crescimento:.2f}")
    st.markdown(
        f"O estado {estado_sel} apresentou crescimento médio diário de {media_crescimento:.2f}% "
        "no período analisado (média simples sobre as variações diárias)."
    )


elif pagina == "Agrupamento (K-Means)":
    st.header("Agrupamento de Estados (K-Means)")

    agr = dados_atuais[["state", "confirmed", "deaths"]].dropna().copy()
    st.markdown("Utilizamos K-Means para agrupar estados com perfis semelhantes em termos de casos e mortes.")

    if len(agr) > 1:
        X = agr[["confirmed", "deaths"]].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        k = st.slider("Número de clusters (k)", min_value=2, max_value=min(6, len(agr)), value=min(3, len(agr)))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = km.fit_predict(Xs)
        agr["cluster"] = clusters

        st.subheader("Tabela de Estados com Cluster")
        st.dataframe(agr.sort_values("cluster").reset_index(drop=True))

        fig, ax = plt.subplots(figsize=(8, 5))
        for c in sorted(agr["cluster"].unique()):
            grupo = agr[agr["cluster"] == c]
            ax.scatter(grupo["confirmed"], grupo["deaths"], label=f"Cluster {c}", s=60)
        ax.set_xlabel("Casos Confirmados")
        ax.set_ylabel("Mortes")
        ax.set_title("Agrupamento de Estados por Casos e Mortes")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            "Interpretação (exemplo):\n\n"
            "- Cluster com menores valores: estados com baixa carga absoluta.\n"
            "- Cluster intermediário: carga média.\n"
            "- Cluster com maiores valores: grandes centros com muitos casos e óbitos (ex.: SP, RJ).\n\n"
            "A interpretação deve considerar população, testagem e reporting."
        )
    else:
        st.warning("Dados insuficientes para realizar o agrupamento.")


elif pagina == "Aprendizagem Supervisionada":
    st.header("Aprendizagem Supervisionada — Regressão (prever óbitos)")

    st.markdown(
        "Aqui usamos os registros agregados por estado (último registro) para treinar um modelo de regressão "
        "que tenta prever o número de óbitos (`deaths`) a partir de casos confirmados (`confirmed`). "
        "Nota: é uma análise exploratória simples — em cenários reais seria necessário enriquecer com variáveis demográficas, tempo desde o surto, etc."
    )

    agr = dados_atuais[["state", "confirmed", "deaths"]].dropna().copy()
    X = agr[["confirmed"]].values  
    y = agr["deaths"].values    


    modelo_selecionado = st.selectbox("Modelo", ["Regressão Linear", "Random Forest (regressor)"])
    test_size = st.slider("Proporção de teste (%)", min_value=10, max_value=40, value=20, step=5)
    random_state = 42

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=random_state)

    if modelo_selecionado == "Regressão Linear":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=random_state)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{r2:.3f}")
    c2.metric("MAE", f"{mae:.2f}")
    c3.metric("RMSE", f"{rmse:.2f}")
    c4.metric("Samples teste", f"{len(y_test)}")

    st.subheader("Gráfico: Valores Reais vs Preditos (conjunto de teste)")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, s=60)
    lims = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
    ax.plot(lims, lims, linestyle="--", color="gray") 
    ax.set_xlabel("Óbitos Reais")
    ax.set_ylabel("Óbitos Preditos")
    ax.set_title(f"Real x Predito — {modelo_selecionado}")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        "Observações:\n\n"
        "- O modelo é simples e serve como demonstração de pipeline de modelagem (treino/teste, métricas e interpretação).\n"
        "- Para melhorar: incluir features adicionais (população, testes por 100k, idade média, tempo desde primeiro caso), "
        "fazer engenharia de features, validação cruzada e análise de resíduos."
    )


st.markdown("---")
st.caption("Projeto Acadêmico — SENAC | 2025")
