import streamlit as st
from langchain_community.llms import OpenAI
from langchain_experimental.agents import create_csv_agent

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
    st.stop()

# ------------------------
# Upload de arquivo
# ------------------------
arquivo = st.file_uploader("📂 Envie um arquivo CSV", type=["csv"])

agent = None

if arquivo is not None:
    try:
        with open("temp.csv", "wb") as f:
            f.write(arquivo.getbuffer())

        agent = create_csv_agent(
            OpenAI(temperature=0, api_key=api_key),
            "temp.csv",
            verbose=False
        )

        st.success("✅ Arquivo carregado e agente criado com sucesso!")

    except Exception as e:
        st.error("❌ Erro ao criar o agente. Verifique o formato do arquivo.")
        st.stop()
else:
    st.warning("📂 Por favor, envie um arquivo para iniciar.")
    st.stop()

# ------------------------
# Inicializa memória
# ------------------------
if "historico" not in st.session_state:
    st.session_state.historico = []

if "resumos" not in st.session_state:
    st.session_state.resumos = []

# ------------------------
# Campo de pergunta
# ------------------------
st.markdown("## 💬 Pergunte ao agente sobre os dados")
pergunta = st.text_input("Digite sua pergunta aqui:")

if pergunta and agent:
    with st.spinner("Analisando com IA..."):
        resposta = agent.invoke(pergunta)
    st.session_state.historico.append((pergunta, resposta))

# ------------------------
# Histórico de perguntas e respostas
# ------------------------
st.subheader("🗂 Histórico de perguntas e respostas")
for i, (q, r) in enumerate(st.session_state.historico):
    st.markdown(f"**{i+1}. Pergunta:** {q}")
    st.markdown(f"**Resposta:** {r}")
    st.markdown("---")

# ------------------------
# Botão: Gerar resumo final
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
# Exportar resumos salvos
# ------------------------
if st.session_state.resumos:
    st.subheader("📁 Histórico de resumos salvos")
    for i, resumo in enumerate(st.session_state.resumos):
        st.markdown(f"**Resumo {i+1}:**")
        st.text_area("", resumo, height=300)
        st.markdown("---")
