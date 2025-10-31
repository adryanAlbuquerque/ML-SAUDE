# ==========================================================
# PREDIÇÃO BASEADA EM DATAS (TIME SERIES)
# ==========================================================
with aba[3]:
    st.header("Predição de Casos Futuros (Time Series)")

    st.markdown("""
    Aqui usamos dados históricos recentes para prever o número de casos nos próximos dias.
    """)

    # Selecionar Estado
    estados_disponiveis = sorted(dados[dados["place_type"] == "state"]["state"].unique())
    estado_pred = st.selectbox("Selecione o estado para previsão:", estados_disponiveis)

    df_pred = dados[(dados["place_type"] == "state") & (dados["state"] == estado_pred)]
    df_pred = df_pred.sort_values("date")
    df_pred = df_pred[["date", "confirmed"]].dropna()

    if len(df_pred) > 30:
        # Preparar dados recentes (últimos 60 dias)
        df_recent = df_pred.tail(60).reset_index(drop=True)
        df_recent["dias"] = range(len(df_recent))

        # Modelo simples de regressão linear no tempo
        X = df_recent[["dias"]]
        y = df_recent["confirmed"]
        modelo = LinearRegression()
        modelo.fit(X, y)

        # Previsão para os próximos 7 dias
        dias_futuros = list(range(len(df_recent), len(df_recent) + 7))
        previsao = modelo.predict(pd.DataFrame({"dias": dias_futuros}))

        # Data correspondente
        ultima_data = df_recent["date"].iloc[-1]
        datas_previstas = pd.date_range(start=ultima_data + pd.Timedelta(days=1), periods=7)

        # Combinar resultados
        df_prev = pd.DataFrame({
            "date": datas_previstas,
            "confirmed_pred": previsao
        })

        st.subheader(f"Previsão de Casos para {estado_pred}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_recent["date"], df_recent["confirmed"], label="Casos Reais", color="steelblue")
        ax.plot(df_prev["date"], df_prev["confirmed_pred"], label="Previsão", color="orange", linestyle="--")
        ax.set_xlabel("Data")
        ax.set_ylabel("Casos Confirmados")
        ax.set_title(f"Evolução e Previsão de Casos em {estado_pred}")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.markdown("🔍 **Observação:** A previsão é baseada em tendência linear simples dos últimos 60 dias.")
    else:
        st.warning("Dados insuficientes para gerar previsão neste estado.")
