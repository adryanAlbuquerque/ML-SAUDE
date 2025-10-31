import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

st.set_page_config(page_title="Análise COVID-19 no Brasil", layout="wide")
sns.set_style("whitegrid")

st.title("Análise de Dados da COVID-19 no Brasil")
st.markdown("""
Aplicação prática desenvolvida para o projeto de **Machine Learning Aplicado à Saúde**.  
Os dados utilizados são públicos e provenientes do repositório [wcota/covid19br](https://github.com/wcota/covid19br).
""")

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

st.sidebar.title("Navegação")
pagina = st.sidebar.radio(
    "Selecione a seção:",
    ["Visão Geral", "Tendências por Estado", "Modelagem Supervisionada"]
)

if pagina == "Visão Geral":
    st.header("Panorama Geral da COVID-19 no Brasil")

    total_casos = int(dados_atuais["casos"].sum())
    total_obitos = int(dados_atuais["obitos"].sum())
    letalidade_geral = (total_obitos / total_casos) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Casos Totais", f"{total_casos:,}".replace(",", "."))
    col2.metric("Óbitos Totais", f"{total_obitos:,}".replace(",", "."))
    col3.metric("Letalidade Média (%)", f"{letalidade_geral:.2f}")

    st.subheader("Casos Confirmados por Estado")
    top_casos = dados_atuais.sort_values("casos", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(y="estado", x="casos", data=top_casos, palette="Blues_r", ax=ax)
    ax.set_xlabel("Casos Confirmados")
    ax.set_ylabel("Estado")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("""
    **Interpretação:**  
    O gráfico mostra o número total de casos confirmados por estado.  
    Estados como São Paulo, Minas Gerais e Rio de Janeiro apresentam os maiores volumes de casos acumulados.
    """)

    st.subheader("Taxa de Letalidade por Estado (%)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="estado", y="letalidade", data=dados_atuais.sort_values("letalidade", ascending=False), palette="Reds_r", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("""
    **Interpretação:**  
    A taxa de letalidade representa o percentual de óbitos entre os casos confirmados.  
    Estados com letalidade mais alta podem indicar subnotificação de casos leves ou limitações no atendimento hospitalar.
    """)

elif pagina == "Tendências por Estado":
    st.header("Evolução Temporal por Estado")

    estado_sel = st.selectbox("Selecione o Estado", sorted(dados["estado"].unique()), index=25)
    df_estado = dados[dados["estado"] == estado_sel].sort_values("data")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(df_estado["data"], df_estado["casos"], color="skyblue", alpha=0.4)
    ax.plot(df_estado["data"], df_estado["casos"], color="blue", linewidth=2)
    ax.set_title(f"Evolução de Casos — {estado_sel}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Casos Confirmados")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("""
    **Interpretação:**  
    O gráfico mostra a evolução acumulada de casos no estado selecionado.  
    Picos ou desacelerações indicam momentos de aumento ou estabilização da transmissão.
    """)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(df_estado["data"], df_estado["obitos"], color="salmon", alpha=0.4)
    ax.plot(df_estado["data"], df_estado["obitos"], color="red", linewidth=2)
    ax.set_title(f"Evolução de Óbitos — {estado_sel}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Óbitos Acumulados")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("""
    **Interpretação:**  
    O crescimento contínuo indica a persistência da mortalidade ao longo do tempo.  
    A redução de inclinação mostra períodos de maior controle da pandemia.
    """)

elif pagina == "Modelagem Supervisionada":
    st.header("Predição de Óbitos com Regressão Linear")

    st.markdown("""
    Esta seção aplica um modelo de **Regressão Linear** para estimar o número de óbitos a partir da quantidade de casos confirmados.  
    O objetivo é demonstrar uma técnica simples de aprendizado supervisionado.
    """)

    X = dados_atuais[["casos"]].values
    y = dados_atuais["obitos"].values

    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        col1, col2, col3 = st.columns(3)
        col1.metric("R²", f"{r2:.2f}")
        col2.metric("MAE", f"{mae:.0f}")
        col3.metric("RMSE", f"{rmse:.0f}")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.regplot(x=y_test, y=y_pred, line_kws={"color": "gray"}, scatter_kws={"s": 80})
        ax.set_xlabel("Óbitos Reais")
        ax.set_ylabel("Óbitos Preditos")
        ax.set_title("Desempenho da Regressão Linear")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        **Interpretação:**  
        A relação linear entre casos e óbitos é evidente — quanto maior o número de casos, maior a tendência de mortes.  
        O modelo supervisionado confirma a correlação forte entre as duas variáveis, sendo útil para previsões em novos cenários.
        """)
    else:
        st.warning("Dados insuficientes para treinar o modelo.")

st.markdown("---")
st.caption("Projeto Acadêmico — Machine Learning Aplicado à Saúde | 2025")
