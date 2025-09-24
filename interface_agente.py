import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.llms import OpenAI

# ------------------------
# Título principal
# ------------------------
st.title("📊 Agente Inteligente com Memória e Gráficos")

# ------------------------
# Entrada de chave de API
# ------------------------
api_key = st.sidebar.text_input("🔑 Digite sua chave da API:", type="password")

if not api_key:
    st.warning("🔒 Digite a chave da API na barra lateral para ativar o agente.")
    st.markdown("---")
    st.subheader("💬 Gerar resumo final do agente")
    st.button("💬 Gerar resumo final do agente", disabled=True)
    st.stop()

# ------------------------
# Upload de arquivo
# ------------------------
arquivo = st.file_uploader("📂 Envie um arquivo CSV ou Excel", type=["csv", "xlsx"])

agent = None
df = None

if arquivo is not None:
    try:
        if arquivo.name.endswith(".csv"):
            df_raw = pd.read_csv(arquivo, header=None)
        else:
            df_raw = pd.read_excel(arquivo, header=None)

        # Detecta se a primeira coluna contém listas como strings
        if isinstance(df_raw.iloc[0, 0], str) and df_raw.shape[1] == 1:
            df = df_raw[0].apply(lambda x: pd.Series(ast.literal_eval(x)))
            df.columns = [f"x{i}" for i in range(df.shape[1])]
        else:
            df = df_raw.copy()

        # 🔒 Blindagem contra erros de estrutura
        st.write("📋 Estrutura do DataFrame:")
        st.dataframe(df)

        st.write("🔍 Colunas detectadas:")
        st.write(df.columns.tolist())

        st.write("🔢 Número de linhas:", len(df))

        if df.empty or df.shape[1] == 0 or df.columns.isnull().any():
            st.error("❌ O DataFrame está vazio ou possui colunas inválidas.")
            st.stop()

        agent = create_pandas_dataframe_agent(
            OpenAI(temperature=0, api_key=api_key),
            df,
            verbose=False
        )

    except Exception as e:
        st.error("❌ Erro ao carregar ou processar o arquivo. Verifique o formato e os dados.")
        st.stop()
else:
    st.warning("Por favor, envie um arquivo para iniciar.")
    st.stop()

# ------------------------
# Inicializa memória
# ------------------------
if "historico" not in st.session_state:
    st.session_state.historico = []

if "resumos" not in st.session_state:
    st.session_state.resumos = []

# ------------------------
# Campo de pergunta visível após agente criado
# ------------------------
st.markdown("## 💬 Pergunte ao agente sobre os dados")
pergunta = st.text_input("Digite sua pergunta aqui:")

if pergunta and agent:
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

# ------------------------
# Botão destacado: Gerar resumo final do agente
# ------------------------
st.markdown("---")
st.subheader("💬 Gerar resumo final do agente")

salvar_limpar = st.checkbox("✅ Grupo salvo e histórico limpo para próxima pergunta.")

if salvar_limpar:
    st.success("✅ Grupo salvo e histórico limpo para próxima pergunta.")

resumo_final_click = st.button("💬 Gerar resumo final do agente", key="resumo_final_btn")
if resumo_final_click:
    if not st.session_state.historico:
        st.info("Histórico vazio — nada para resumir.")
    else:
        historico_texto = ""
        for q, r in st.session_state.historico:
            historico_texto += f"Pergunta: {q}\nResposta: {r}\n\n"

        prompt_resumo_final = f"""
        Você é um analista de dados com habilidades avançadas de linguagem natural.
        Seu objetivo é gerar um RESUMO FINAL detalhado e contextual do histórico de perguntas e respostas abaixo.

        Apresente o conteúdo em tópicos com emojis para facilitar a leitura visual, como ✅, 📊, 🔍, 💡, entre outros.

        O resumo deve conter:
        - Principais temas abordados nas perguntas
        - Análises realizadas e seus resultados
        - Padrões ou correlações identificadas nos dados
        - Conclusões relevantes e possíveis implicações
        - Sugestões de próximos passos ou investigações futuras

        Histórico:
        {historico_texto}
        """

        with st.spinner("Gerando resumo final com IA..."):
            resumo_final = agent.invoke(prompt_resumo_final)

        st.text_area("🧾 Resumo final gerado pelo agente:", resumo_final, height=400)

        if salvar_limpar:
            st.session_state.resumos.append(resumo_final)
            st.session_state.historico = []
            st.success("✅ Grupo salvo e histórico limpo para próxima pergunta.")

# ------------------------
# Exportar resumos salvos (opcional)
# ------------------------
if st.session_state.resumos:
    st.subheader("📁 Histórico de resumos salvos")
    for i, resumo in enumerate(st.session_state.resumos):
        st.markdown(f"**Resumo {i+1}:**")
        st.text_area("", resumo, height=300)
        st.markdown("---")
