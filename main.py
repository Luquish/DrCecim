import os
import json
import time

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Load config
working_dir = os.path.dirname(os.path.abspath(__file__))

config_data = json.load(open(os.path.join(working_dir, "config.json")))

groq_api_key = config_data["GROQ_API_KEY"]

os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize embeddings
def startup_vectorstore():
    persist_directory = os.path.join(working_dir, "vector_db_dir")
    # Using all-MiniLM-L6-v2 model which works well for domain-specific knowledge
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    import chromadb
    from chromadb.config import Settings
    
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True
        )
    )
    
    vectorstore = Chroma(
        client=client,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore

def chat_chain(vectorstore):
    llm = ChatGroq(model_name="mixtral-8x7b-32768",
                   temperature=0.2)  # Slight temperature increase for more natural responses
    
    # Basic retriever
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve more documents for better context
    )
    
    # Add contextual compression to filter irrelevant parts
    compressor = LLMChainExtractor.from_llm(llm)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    # Using window memory to limit context size
    memory = ConversationBufferWindowMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True,
        k=5  # Keep only last 5 exchanges
    )

    qa_template = """Sos DrCecim, un asistente virtual especializado en proporcionar información sobre la Facultad de Medicina de la Universidad de Buenos Aires (UBA).

    Directrices:
        - Responde preguntas orientativas y administrativas sobre la facultad (inscripciones, nombres de profesores, métodos de aprobación, trámites, etc.).
        - No brindes información médica ni consejos sobre salud bajo ninguna circunstancia.
        - Usa un tono amigable, cercano y lo más argentino posible. Incorpora expresiones coloquiales y de lunfardo como "dale", "de una", "bancame", "laburar" y "re" para darle un toque auténtico.
        - Usa emojis para hacer la conversación más dinámica 😊👩‍⚕️🏥📚.
        - Si la pregunta no tiene respuesta en la base de datos, indica amablemente que no podés ayudar y deriva al usuario al centro de estudiantes: 📩 drcecim@uba.com.
        - Si la pregunta es ambigua o incompleta, solicita más detalles antes de responder (por ejemplo: "¿Podrías especificar un poco más a qué te referís?").
        - Recuerda el historial de conversación para hacer la interacción más fluida y evitar respuestas repetitivas.
        - Al final de cada interacción, si hay 5 minutos de inactividad, pregunta si la respuesta fue útil para recibir feedback.

    Estructura de la Respuesta:
        - Explica de manera clara y sencilla.
        - Si la información proviene de la base de datos, menciona la fuente usando el formato: [Fuente: nombre_documento]. Si se usan múltiples fuentes, sepáralas por comas.
        - Organiza la información en párrafos cortos para facilitar la lectura.
        - Si la respuesta puede ser respondida con un punteo(en el caso de poder implementar emojis para representar lo dicho, agregar uno al final del punteo)
        - Si la consulta se sale del ámbito (por ejemplo, sobre consejos de salud), informa que no podés ayudar en ese tema y sugerí consultar a un profesional.
        - Solo saluda en el primer mensaje. Para saludar podes usar "Como va?", "Como andás?" "Buenas bueans", "Hola"

    Ejemplos de respuesta:

    Pregunta: Che, ¿me bancás y me decís cuándo arrancan las inscripciones para el CBC?
    Respuesta: ¡Buenas buenas! 👋 Las inscripciones para el CBC de Medicina arrancan el 15 de noviembre y terminan el 15 de diciembre. Recordá llevar tu DNI y la constancia de título. ¡Bancate y no te quedes afuera! 📅✏️ [Fuente: Calendario_Académico_2023]

    Pregunta: Dale, contame re bien cómo labura el sistema de exámenes en la facultad.
    Respuesta: El sistema de exámenes en la Facultad de Medicina funciona así: primero, te inscribís en la materia y luego te asignan una mesa examinadora. Si tenés dudas, no te preocupés, ¡estoy para darte una mano! 😊 [Fuente: Normativa_Facultad]

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
        chain_type="stuff",  # Using stuff for shorter documents
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

if "last_interaction_time" not in st.session_state:
    st.session_state.last_interaction_time = time.time()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check for inactivity (5 minutes)
current_time = time.time()
if (current_time - st.session_state.last_interaction_time > 300 and 
    len(st.session_state.messages) > 0 and 
    st.session_state.messages[-1]["role"] != "assistant_feedback"):
    with st.chat_message("assistant"):
        feedback_message = "¿Te fue útil mi respuesta? ¿Necesitás algo más? 😊"
        st.markdown(feedback_message)
        st.session_state.messages.append({"role": "assistant_feedback", "content": feedback_message})

user_input = st.chat_input("Consultale a DrCecim")

if user_input:
    st.session_state.last_interaction_time = time.time()
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("DrCecim está pensando..."):
            response = st.session_state.conversational_chain({
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })
            assistant_message = response["answer"]
            
            # Store source documents for reference
            sources = [doc.metadata.get("source", "desconocido") for doc in response.get("source_documents", [])]
            
            st.markdown(assistant_message)
            st.session_state.messages.append({"role": "assistant", "content": assistant_message})
            
            # Update conversation history for the chain
            st.session_state.chat_history.append((user_input, assistant_message))