import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.llms import OpenAI

# ------------------------
# Título principal
# ------------------------
st.title("📊 Agente Inteligente para Qualquer Arquivo")

# ------------------------
# Entrada de chave de API
# ------------------------
api_key = st.sidebar.text_input("🔑 Digite sua chave da API:", type="password")

if not api_key:
    st.warning("🔒 Digite a chave da API na barra lateral para ativar o agente.")
    st.stop()

# ------------------------
# Upload de arquivo
# ------------------------
arquivo = st.file_uploader("📂 Envie um arquivo CSV ou Excel", type=["csv", "xlsx"])

if arquivo is not None:
    if arquivo.name.endswith(".csv"):
        df = pd.read_csv(arquivo)
    else:
        df = pd.read_excel(arquivo)

    agent = create_pandas_dataframe_agent(
        OpenAI(temperature=0, api_key=api_key),
        df,
        verbose=False
    )
else:
    st.warning("Por favor, envie um arquivo para iniciar.")
    st.stop()

# ------------------------
# Histórico e memória
# ------------------------
if "historico" not in st.session_state:
    st.session_state.historico = []

# ------------------------
# Entrada de pergunta
# ------------------------
pergunta = st.text_input("💬 Faça uma pergunta sobre os dados:")

if pergunta:
    with st.spinner("Analisando com IA..."):
        resposta = agent.invoke(pergunta)
    st.session_state.historico.append((pergunta, resposta))

# ------------------------
# Exibir histórico
# ------------------------
st.subheader("🗂 Histórico de perguntas e respostas")
for i, (q, r) in enumerate(st.session_state.historico):
    st.markdown(f"**{i+1}. Pergunta:** {q}")
    st.markdown(f"**Resposta:** {r}")
    st.markdown("---")

# ------------------------
# Gráficos automáticos
# ------------------------
if pergunta and "gráfico" in pergunta.lower():
    if "amount" in pergunta.lower() and "Amount" in df.columns:
        st.subheader("📊 Gráfico da coluna Amount")
        fig, ax = plt.subplots()
        sns.histplot(df['Amount'], bins=50, ax=ax)
        st.pyplot(fig)
    elif "tempo" in pergunta.lower() and "fraude" in pergunta.lower() and "Time" in df.columns and "Class" in df.columns:
        st.subheader("📊 Fraudes ao longo do tempo")
        fig, ax = plt.subplots()
        sns.lineplot(data=df[df['Class'] == 1], x='Time', y='Amount', ax=ax)
        st.pyplot(fig)
