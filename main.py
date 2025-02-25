import os
import json

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Load config
working_dir = os.path.dirname(os.path.abspath(__file__))

config_data = json.load(open(os.path.join(working_dir, "config.json")))

groq_api_key = config_data["GROQ_API_KEY"]

os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize embeddings
def startup_vectorstore():
    persist_directory = os.path.join(working_dir, "vector_db_dir")
    embeddings = HuggingFaceEmbeddings()
    
    import chromadb
    
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
    
    client = chromadb.PersistentClient(path=persist_directory)
    
    vectorstore = Chroma(
        client=client,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore

def chat_chain(vectorstore):
    llm = ChatGroq(model_name="llama3-8b-8192",
                   temperature=0)
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )

    qa_template = """Eres DrCecim, un asistente médico virtual especializado en proporcionar información acerca de la Facultad de Mediciona de la Universidad de Buenos Aires. 
        Debes:
        - Responder preguntas de manera clara y comprensible
        - Citar fuentes el pdf de donde sacaste la información
        - Ser siempre amigable y compañero
        - Mantener un tono profesional pero amigable

        Contexto: {context}
        Historial del chat: {chat_history}
        Pregunta humana: {question}
        Respuesta del asistente:"""

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=qa_template
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        verbose=True,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return chain

st.set_page_config(page_title="🥼 DrCecim Chatbot Demo",
                   page_icon="🥼",
                   layout="centered")

st.title("DrCecim Chatbot Demo")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = startup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Consulale a DrCecim")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversational_chain({
            "question": user_input,
            "chat_history": st.session_state.chat_history
        })
        assistant_message = response["answer"]
        st.markdown(assistant_message)
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
        
        
