import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------
# TÍTULO E INTRODUÇÃO
# -----------------------------------------------
st.title("Machine Learning Aplicado à Saúde - Câncer de Mama")
st.markdown("""
Este aplicativo demonstra o uso de **técnicas de aprendizado de máquina supervisionadas e não supervisionadas**
em um conjunto de dados reais sobre **câncer de mama**.

O objetivo é mostrar como é possível **coletar, tratar, modelar e interpretar dados médicos**
de forma acessível e visual.
""")

# -----------------------------------------------
# CARREGAR O DATASET
# -----------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

df = X.copy()
df["diagnóstico"] = y.map({0: "Maligno", 1: "Benigno"})

st.header("1. Coleta e visualização dos dados")
st.write("O conjunto de dados contém informações sobre células tumorais de pacientes.")
st.dataframe(df.head())

st.write("Tamanho do conjunto de dados:", df.shape)

# -----------------------------------------------
# ANÁLISE EXPLORATÓRIA
# -----------------------------------------------
st.header("2. Análise exploratória")

st.subheader("Distribuição das classes (diagnósticos)")
st.bar_chart(df["diagnóstico"].value_counts())

st.subheader("Estatísticas descritivas")
st.write(df.describe())

st.subheader("Correlação entre as variáveis numéricas")
corr = X.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# -----------------------------------------------
# FILTRO SIMPLES
# -----------------------------------------------
st.header("3. Filtro de dados")
st.write("Use o filtro abaixo para selecionar apenas os exames com determinado valor mínimo de raio médio.")

valor_raio = st.slider(
    "Raio médio mínimo (mean radius)",
    float(X["mean radius"].min()),
    float(X["mean radius"].max()),
    float(X["mean radius"].mean())
)

df_filtrado = df[df["mean radius"] >= valor_raio]
st.write(f"Total de registros após filtro: {df_filtrado.shape[0]}")
st.dataframe(df_filtrado.head())

# -----------------------------------------------
# TRATAMENTO DOS DADOS
# -----------------------------------------------
st.header("4. Tratamento dos dados")

st.write("""
Antes de treinar o modelo, é necessário padronizar os dados.
Isso garante que todas as variáveis tenham a mesma escala, 
evitando que uma característica influencie mais que as outras.
""")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------
# MODELO SUPERVISIONADO - CLASSIFICAÇÃO
# -----------------------------------------------
st.header("5. Modelo supervisionado (Regressão Logística)")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
st.write(f"Acurácia do modelo: **{acuracia * 100:.2f}%**")

st.subheader("Matriz de confusão")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")
st.pyplot(fig)

st.subheader("Relatório de classificação")
st.text(classification_report(y_test, y_pred, target_names=["Maligno", "Benigno"]))

# -----------------------------------------------
# MODELO NÃO SUPERVISIONADO - AGRUPAMENTO
# -----------------------------------------------
st.header("6. Modelo não supervisionado (K-Means)")

st.write("""
O K-Means é uma técnica de agrupamento que tenta separar os dados em grupos (clusters)
com base em suas semelhanças. Aqui, usamos 2 grupos, já que há dois tipos de diagnósticos.
""")

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters

fig, ax = plt.subplots()
sns.scatterplot(
    x=df["mean radius"],
    y=df["mean texture"],
    hue=df["Cluster"],
    palette="viridis",
    ax=ax
)
ax.set_title("Visualização dos agrupamentos (K-Means)")
st.pyplot(fig)

st.write("Comparação entre o cluster atribuído e o diagnóstico real:")
st.table(df.groupby(["Cluster", "diagnóstico"]).size())

# -----------------------------------------------
# INTERPRETAÇÃO FINAL
# -----------------------------------------------
st.header("7. Interpretação dos resultados")
st.markdown("""
- O modelo supervisionado (Regressão Logística) apresentou **alta acurácia**, mostrando que as características das células são bons indicadores do tipo de tumor.  
- O modelo não supervisionado (K-Means) conseguiu identificar padrões nos dados sem usar os rótulos, agrupando de forma parecida aos diagnósticos reais.  
- A análise exploratória mostrou forte correlação entre o tamanho, área e perímetro das células — indicando que tumores malignos geralmente possuem **núcleos maiores e mais irregulares**.  
""")

st.success("Análise concluída com sucesso.")
