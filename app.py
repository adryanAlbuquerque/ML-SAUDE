import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="Análise de COVID-19 no Brasil", layout="wide")

st.title("Machine Learning Aplicado à Saúde: COVID-19 no Brasil")
st.markdown("""
Este projeto aplica técnicas de **Aprendizagem Supervisionada** sobre dados reais da COVID-19 no Brasil.  
O objetivo é coletar, tratar, modelar e interpretar dados de forma clara e acessível.
""")

@st.cache_data
def carregar_dados():
    url = "https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv"
    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"state": "estado", "newDeaths": "novos_obitos", "newCases": "novos_casos", 
                            "deaths": "obitos", "totalCases": "casos"})
    df = df[["date", "estado", "casos", "obitos", "novos_casos", "novos_obitos"]]
    df = df[df["estado"] != "TOTAL"]
    return df

dados = carregar_dados()
estados = sorted(dados["estado"].unique())
pagina = st.sidebar.radio("Navegação", ["Exploração dos Dados", "Análise por Estado", "Modelagem Supervisionada"])

if pagina == "Exploração dos Dados":
    st.header("Exploração dos Dados")
    st.markdown("Nesta seção, apresentamos uma visão geral dos casos e óbitos confirmados de COVID-19 no Brasil.")

    dados_atuais = dados.groupby("estado")[["casos", "obitos"]].max().reset_index()
    dados_atuais["letalidade"] = (dados_atuais["obitos"] / dados_atuais["casos"]) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dados_atuais.sort_values("casos", ascending=False), x="estado", y="casos", ax=ax)
    ax.set_title("Casos Confirmados por Estado")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Interpretação:** Os estados com maior número de casos são São Paulo, Minas Gerais e Paraná, refletindo suas populações e densidade urbana.")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dados_atuais.sort_values("obitos", ascending=False), x="estado", y="obitos", ax=ax, color="salmon")
    ax.set_title("Óbitos Confirmados por Estado")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Interpretação:** A distribuição de óbitos segue o padrão dos casos, com maiores números em estados de alta população e circulação de pessoas.")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dados_atuais.sort_values("letalidade", ascending=False), x="estado", y="letalidade", ax=ax, color="gray")
    ax.set_title("Taxa de Letalidade (%) por Estado")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Interpretação:** Estados do Norte e Nordeste tendem a apresentar maior taxa de letalidade, possivelmente associada à menor infraestrutura hospitalar.")

elif pagina == "Análise por Estado":
    st.header("Análise Individual por Estado")
    estado_selecionado = st.selectbox("Selecione o estado:", estados)
    df_estado = dados[dados["estado"] == estado_selecionado]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_estado["date"], df_estado["casos"], label="Casos", color="blue")
    ax.plot(df_estado["date"], df_estado["obitos"], label="Óbitos", color="red")
    ax.legend()
    ax.set_xlabel("Data")
    ax.set_ylabel("Quantidade")
    ax.set_title(f"Evolução da COVID-19 em {estado_selecionado}")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Interpretação:** O gráfico mostra a evolução acumulada de casos e óbitos ao longo do tempo para o estado selecionado. A distância entre as curvas reflete a taxa de letalidade local.")

    taxa_crescimento = df_estado["casos"].pct_change().mean() * 100
    st.metric("Crescimento Médio Diário de Casos (%)", f"{taxa_crescimento:.2f}")

elif pagina == "Modelagem Supervisionada":
    st.header("Predição de Óbitos com Regressão Linear")

    st.markdown("""
    Esta seção aplica um modelo de **Regressão Linear** para estimar o número de óbitos a partir da quantidade de casos confirmados.  
    O objetivo é demonstrar uma técnica simples de aprendizado supervisionado.
    """)

    dados_atuais = dados.groupby("estado")[["casos", "obitos"]].max().reset_index()
    X = dados_atuais[["casos"]].values
    y = dados_atuais["obitos"].values

    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

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
        A relação linear entre casos e óbitos é clara — quanto maior o número de casos, maior a tendência de mortes.  
        O modelo supervisionado demonstra uma correlação forte entre as duas variáveis, sendo útil para estimativas e análises preditivas simples.
        """)
    else:
        st.warning("Dados insuficientes para treinar o modelo.")
