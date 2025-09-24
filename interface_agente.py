import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from langchain_experimental.agents import create_csv_agent
from langchain_community.llms import OpenAI  # Pacote atualizado e estável

# ------------------------
# Título principal
# ------------------------
st.title("📊 Agente Inteligente com Memória e Gráficos")

# ------------------------
# Entrada de chave de API
# ------------------------
api_key = st.sidebar.text_input("Digite sua chave da API (OpenAI, Gemini, etc.):", type="password")

# ------------------------
# Exibe botão visual mesmo sem chave
# ------------------------
if not api_key:
    st.warning("🔒 Digite a chave da API na barra lateral para ativar o agente.")
    st.markdown("---")
    st.subheader("💬 Gerar resumo final do agente")
    st.button("💬 Gerar resumo final do agente", disabled=True)
    st.stop()

# ------------------------
# Upload de arquivo e criação do agente
# ------------------------
arquivo = st.file_uploader("📂 Envie o arquivo Excel ou CSV para análise", type=["xlsx", "csv"])

if arquivo is not None:
    if arquivo.name.endswith(".csv"):
        try:
            df = pd.read_csv(arquivo)
            agent = create_csv_agent(
                OpenAI(temperature=0, api_key=api_key),
                arquivo,
                verbose=False,
                allow_dangerous_code=True
            )
        except pd.errors.EmptyDataError:
            st.error("❌ O arquivo CSV está vazio ou mal formatado.")
            st.stop()
    else:
        df = pd.read_excel(arquivo)
        agent = None
else:
    st.warning("Por favor, envie um arquivo CSV ou Excel para iniciar a análise.")
    st.stop()

# ------------------------
# Inicializa memória
# ------------------------
if "historico" not in st.session_state:
    st.session_state.historico = []

if "resumos" not in st.session_state:
    st.session_state.resumos = []

# ------------------------
# Entrada de pergunta
# ------------------------
pergunta = st.text_input("Digite sua pergunta sobre os dados:")

if pergunta and agent:
    with st.spinner("Analisando com IA..."):
        resposta = agent.invoke(pergunta)
    st.session_state.historico.append((pergunta, resposta))
elif pergunta and not agent:
    st.info("Este tipo de arquivo não permite análise com IA. Use um CSV para ativar o agente.")

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
# Resumo com IA contextual em bullet points com emojis
# ------------------------
with st.expander("🧾 Resumo Automático com IA"):
    st.write("Clique no botão para o agente gerar um resumo contextual em tópicos de toda a conversa.")
    resumo_ia_click = st.button("Gerar Resumo com IA", key="resumo_ia_btn")
    if resumo_ia_click and agent:
        historico_texto = ""
        for q, r in st.session_state.historico:
            historico_texto += f"Pergunta: {q}\nResposta: {r}\n\n"

        prompt_resumo = f"""
        Gere um resumo claro e organizado do histórico de perguntas e respostas abaixo.
        Apresente o resultado em tópicos (bullet points) e utilize emojis para dar impacto visual,
        como ✅, 📊, 🔹, entre outros. Destaque os principais pontos analisados,
        insights encontrados e conclusões relevantes.

        Histórico:
        {historico_texto}
        """

        with st.spinner("Gerando resumo com IA..."):
            resumo_ia = agent.invoke(prompt_resumo)

        st.text_area("Resumo gerado pelo agente (em tópicos com emojis):", resumo_ia, height=400)
    elif resumo_ia_click and not agent:
        st.info("Resumo com IA indisponível para arquivos Excel. Use CSV para ativar o agente.")

# ------------------------
# Botão destacado igual à imagem: Gerar resumo final do agente
# ------------------------
st.markdown("---")
st.subheader("💬 Gerar resumo final do agente")

salvar_limpar = st.checkbox("✅ Grupo salvo e histórico limpo para próxima pergunta.")

if salvar_limpar:
    st.success("✅ Grupo salvo e histórico limpo para próxima pergunta.")

resumo_final_click = st.button("💬 Gerar resumo final do agente", key="resumo_final_btn")
if resumo_final_click and agent:
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

        Se houver menção a gráficos, inclua uma breve descrição do que foi visualizado.

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
elif resumo_final_click and not agent:
    st.info("Resumo com IA indisponível para arquivos Excel. Use CSV para ativar o agente.")

# ------------------------
# Exportar resumos salvos (opcional)
# ------------------------
if st.session_state.resumos:
    st.subheader("📁 Histórico de resumos salvos")
    for i, resumo in enumerate(st.session_state.resumos):
        st.markdown(f"**Resumo {i+1}:**")
        st.text_area("", resumo, height=300)
        st.markdown("---")
