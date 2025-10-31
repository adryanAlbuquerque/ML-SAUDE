import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- TÍTULO ---
st.title("🩺 Machine Learning Aplicado à Saúde - Câncer de Mama")
st.write("Este aplicativo demonstra o uso de técnicas supervisionadas e não supervisionadas em um conjunto de dados de saúde.")

# --- CARREGAR DATASET ---
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

df = X.copy()
df['target'] = y.map({0: 'maligno', 1: 'benigno'})

st.subheader("📊 Visualização dos Dados")
st.dataframe(df.head())

# --- FILTRO SIMPLES ---
st.sidebar.header("🔎 Filtro de Dados")
mean_radius = st.sidebar.slider("Filtrar por raio médio (mean radius)", float(X['mean radius'].min()), float(X['mean radius'].max()), float(X['mean radius'].mean()))
filtered_df = df[df['mean radius'] >= mean_radius]
st.write(f"Mostrando {filtered_df.shape[0]} registros com raio médio ≥ {mean_radius:.2f}")
st.dataframe(filtered_df.head())

# --- TREINAMENTO SUPERVISIONADO ---
st.subheader("🤖 Modelo Supervisionado - Regressão Logística")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

st.write(f"Acurácia do modelo: **{acc*100:.2f}%**")

# --- MODELO NÃO SUPERVISIONADO ---
st.subheader("🧩 Modelo Não Supervisionado - KMeans")

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

df['cluster'] = clusters

fig, ax = plt.subplots()
ax.scatter(df['mean radius'], df['mean texture'], c=df['cluster'], cmap='viridis', alpha=0.6)
ax.set_xlabel("mean radius")
ax.set_ylabel("mean texture")
ax.set_title("Agrupamento KMeans (2 clusters)")
st.pyplot(fig)

st.success("✅ Aplicação executada com sucesso!")
