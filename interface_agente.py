import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI

# Carrega a chave da API
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Interface lateral para entrada da chave da API
st.sidebar.title("Configuração")
api_key_input = st.sidebar.text_input("Digite sua chave da API:", type="password")

if api_key_input:
    openai_api_key = api_key_input

st.title("Agente Inteligente com Memória e Gráficos")

# Upload do arquivo CSV ou XLSX
arquivo = st.file_uploader("Envie seu arquivo CSV ou XLSX:", type=["csv", "xlsx"])

if arquivo is not None and openai_api_key:
    # Leitura do arquivo
    if arquivo.name.endswith(".csv"):
        df = pd.read_csv(arquivo)
    else:
        df = pd.read_excel(arquivo)

    # Criação do agente — com handle_parsing_errors=True adicionado
    agent = create_pandas_dataframe_agent(
        OpenAI(temperature=0, openai_api_key=openai_api_key),
        df,
        verbose=False,
        handle_parsing_errors=True,  # ← Aqui está o ajuste solicitado
        allow_dangerous_code=True
    )

    # Inicializa variáveis de sessão
    if "historico" not in st.session_state:
        st.session_state.historico = []

    if "grupos_salvos" not in st.session_state:
        st.session_state.grupos_salvos = []

    pergunta = st.text_input("Digite sua pergunta sobre o conjunto de dados:")

    if pergunta:
        try:
            resposta = agent.run(pergunta)
        except Exception as e:
            resposta = f"Erro ao processar: {str(e)}"

        st.session_state.historico.append((pergunta, resposta))

        st.subheader(f"Resposta para: {pergunta}")
        st.write(resposta)

        if "distribuição" in pergunta.lower() or "gráfico" in pergunta.lower():
            colunas_numericas = df.select_dtypes(include=["float64", "int64"]).columns
            for coluna in colunas_numericas:
                if coluna.lower() in pergunta.lower():
                    st.subheader(f"Distribuição da variável: {coluna}")
                    fig, ax = plt.subplots()
                    sns.histplot(df[coluna], kde=True, ax=ax)
                    st.pyplot(fig)
                    break

        st.subheader("Histórico atual")
        for i, (p, r) in enumerate(st.session_state.historico):
            st.markdown(f"**{i+1}. Pergunta:** {p}")
            st.markdown(f"**Resposta:** {r}")

        st.session_state.grupos_salvos.append(st.session_state.historico.copy())
        st.session_state.historico = []
        st.success("Grupo salvo. Pronto para a próxima pergunta.")

    if st.button("Gerar resumo final do agente"):
        resumo = ""
        for grupo in st.session_state.grupos_salvos:
            for pergunta, resposta in grupo:
                resumo += f"- {pergunta}\n  → {resposta}\n\n"
        st.subheader("Resumo Final")
        st.text_area("Resumo gerado:", resumo.strip(), height=400)

else:
    st.warning("Digite sua chave da API na barra lateral e envie um arquivo para ativar o agente.")
