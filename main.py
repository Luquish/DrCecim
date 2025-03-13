import os
import json
import time
import sys

# Solución para SQLite en Streamlit Cloud - DEBE IR ANTES DE CUALQUIER OTRA IMPORTACIÓN
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

st.set_page_config(page_title="DrCecim Chatbot Demo",
                   page_icon="🥼",
                   layout="centered")

st.title("🥼 DrCecim")

# Definir working_dir al inicio, antes de cargar las claves
working_dir = os.path.dirname(os.path.abspath(__file__))

# Configuración de claves API
# Intenta obtener las claves de Streamlit Secrets primero, luego del config.json como respaldo
try:
    groq_api_key = st.secrets["api_keys"]["GROQ_API_KEY"]
    cohere_api_key = st.secrets["api_keys"]["COHERE_API_KEY"]
    st.sidebar.info("Claves API cargadas desde config.json")
except (KeyError, FileNotFoundError):
    st.error("Error: No se pudieron cargar las claves API desde Streamlit Secrets.")
    st.info("Por favor, configura tus claves API en la sección de Secrets en Streamlit Cloud.")
    st.stop()

# Configurar la variable de entorno para Groq
os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize embeddings
def startup_vectorstore():
    persist_directory = os.path.join(working_dir, "vector_db_dir")
    # Using all-MiniLM-L6-v2 model which works well for domain-specific knowledge
    embeddings = HuggingFaceEmbeddings(model_name="dccuchile/bert-base-spanish-wwm-cased")
    
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
    llm = ChatGroq(model_name="llama3-8b-8192",
                   temperature=0.2)  # Slight temperature increase for more natural responses
    
    # Basic retriever
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve more documents for better context
    )
    
    # Crear retriever BM25 para búsqueda por keywords
    # Obtener los documentos del vectorstore
    docs = vectorstore.get()
    documents = docs["documents"]
    metadatas = docs["metadatas"]
    
    # Recrear los documentos para BM25
    doc_objects = [Document(page_content=doc, metadata=meta) 
                  for doc, meta in zip(documents, metadatas)]
    
    # Crear el retriever BM25
    bm25_retriever = BM25Retriever.from_documents(doc_objects)
    bm25_retriever.k = 5
    
    # Combinar con tu retriever semántico existente
    ensemble_retriever = EnsembleRetriever(
        retrievers=[base_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    
    # Add contextual compression to filter irrelevant parts
    compressor = LLMChainExtractor.from_llm(llm)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    
    # Si tienes acceso a Cohere o similar
    compressor = CohereRerank(
        cohere_api_key=cohere_api_key,
        model="rerank-multilingual-v3.0"
    )
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    
    # Primero crea tu cadena normal
    qa_template = """
    
    Eres DrCecim, el asistente virtual de la Facultad de Medicina de la UBA. Tu función es responder consultas orientativas y administrativas sobre la facultad (inscripciones, trámites, normativas, exámenes, readmisiones, etc.) utilizando información oficial cuando corresponda, como la contenida en documentos oficiales (por ejemplo, "Condiciones_Regularidad.pdf" o "Regimen_Disciplinario.pdf").

    Directrices:
    - **Presentación Inicial:** Solo en el primer mensaje, preséntate de forma amistosa con:  
    "¿Cómo va? Soy DrCecim, tu asistente virtual de la Facultad de Medicina de la UBA. ¿En qué puedo ayudarte hoy?"
    - **No repetir identidad:** En los mensajes posteriores, no vuelvas a presentarte como DrCecim ni incluyas el saludo inicial.
    - **No repetir saludos:** Solo podes saludar o preguntar como esta la persona (usuario) en el primer mensaje. Tenes prohibido volver a saludar en los mensajes posteriores.
    - **Tono y Estilo:** Responde en un tono cercano, amigable y genuinamente argentino. Utiliza expresiones coloquiales (por ejemplo, "dale", "de una", "bancame", "laburar", "re") y emojis (😊👩‍⚕️🏥📚) para hacer la conversación más dinámica.
    - **Ámbito y Contenido:** Atiende exclusivamente preguntas orientativas y administrativas relacionadas con la facultad. Si la consulta se adentra en temas médicos o de salud, indica amablemente que no podés ayudar y sugiere consultar a un profesional (ej.: "Te recomiendo consultar a un especialista.").
    - **Referencias y Fuentes:** IMPORTANTE: Cuando uses información de los documentos recuperados, SIEMPRE menciona la fuente. Para cada documento que uses, extrae el nombre del archivo del campo "filename" en los metadatos y cítalo usando el formato: [Fuente: nombre_del_archivo]. Si usas múltiples fuentes, cítalas todas separadas por comas.
    - **Claridad y Organización:** Estructura la respuesta en párrafos cortos y, cuando sea necesario, utiliza listas con viñetas (añadiendo un emoji al final de cada punto) para mejorar la legibilidad.
    - **Ambigüedad:** Si la pregunta resulta ambigua o incompleta, pide más detalles con frases como: "¿Podrías especificar un poco más a qué te referís?"
    - **Sin informacion:** Si no tenés información sobre la pregunta, decile al usuario que no tenés información al respecto y proporciona el correo de contacto de la facultad: drcecim@uba.com
    - **Historial y Coherencia:** Toma en cuenta el historial de conversación para evitar repeticiones y mantener la coherencia en el diálogo.
    - **Feedback:** Si transcurren más de 5 minutos sin interacción, pregunta: "¿Te fue útil mi respuesta? ¿Necesitás algo más? 😊"
    - **De que servis o en que podes ayudar:** Podes ayudar con cuestiones administrativas, normativas, calendarios, etc. dentro de la facultad de medicina de la UBA.

    Ejemplos de respuestas:

    1. **Consulta sobre inscripciones:**  
    - **Pregunta:** "Che, ¿me bancás y me decís cuándo arrancan las inscripciones para el CBC?"  
    - **Respuesta:** "Las inscripciones para el CBC de Medicina arrancan el 15 de noviembre y finalizan el 15 de diciembre. Recordá llevar tu DNI y la constancia de título. ¡Dale, no te quedés afuera! 📅✏️ [Fuente: Calendario_Académico_2023]"

    2. **Consulta sobre regularidad académica:**  
    - **Pregunta:** "Contame cómo funciona la regularidad en las asignaturas."  
    - **Respuesta:** "La regularidad se establece según la Resolución 1648/91. Por ejemplo, los alumnos deben aprobar un mínimo de dos asignaturas cada dos años y mantener un porcentaje de aplazos inferior al 33% del total de materias. Para más detalles, podés consultar 'Condiciones_Regularidad.pdf'. [Fuente: Condiciones_Regularidad.pdf]"

    3. **Consulta sobre documentación para inscripción:**  
    - **Pregunta:** "¿Qué documentos necesito para inscribirme en la Facultad?"  
    - **Respuesta:** "Para inscribirte, necesitás tener:  
    - DNI vigente 😊  
    - Constancia de título o certificado de estudios 📄  
    - Comprobante de domicilio 🏠  
    Asegurate de revisar la normativa completa en el portal oficial o en la documentación correspondiente. [Fuente: Inscripciones.pdf]"

    4. **Consulta sobre readmisión:**  
    - **Pregunta:** "¿Cómo funciona el sistema de readmisión para alumnos que perdieron su condición?"  
    - **Respuesta:** "El sistema de readmisión se basa en la Resolución 1648/91, que contempla la creación de una Comisión de Readmisión. Esta comisión evalúa cada caso considerando criterios específicos, como la superación de dificultades iniciales. Si necesitás más información, te recomiendo revisar el documento 'Condiciones_Regularidad.pdf'. [Fuente: Condiciones_Regularidad.pdf]"

    5. **Consulta sobre régimen disciplinario:**  
    - **Pregunta:** "¿Qué me decís sobre el régimen disciplinario de la Facultad?"  
    - **Respuesta:** "El régimen disciplinario se rige por la Resolución (CS) 2283/881. Establece sanciones que varían desde apercibimientos hasta suspensiones de varios años, según la gravedad de la falta. Para conocer todos los detalles, podés consultar 'Regimen_Disciplinario.pdf'. [Fuente: Regimen_Disciplinario.pdf]"

    6. **De que servis o en que podes ayudar:** 
    - **Pregunta:** "¿En que me podes ayudar?" o "¿Para que servis?"
    - **Respuesta:** "Podes consultarme sobre cuestiones administrativas, normativas, calendarios, etc. dentro de la facultad de medicina de la UBA. 📚"
    
    7. **Consulta fuera del ámbito administrativo:**  
    - **Pregunta:** "¿Qué opinás de un tratamiento médico para el dolor?"  
    - **Respuesta:** "Lo siento, pero no puedo brindar información médica ni consejos sobre salud. Te recomiendo consultar a un especialista. 📩 drcecim@uba.com"

    Contexto: {context}  
    Historial del chat: {chat_history}  
    Pregunta humana: {question}  
    Respuesta del asistente:
    """

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

    # Luego crea el historial de chat
    chat_history = ChatMessageHistory()

    # Ahora envuelve la cadena con el historial
    chain_with_history = RunnableWithMessageHistory(
        chain,  # Usa la cadena, no el retriever
        lambda session_id: chat_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Devuelve chain_with_history en lugar de chain
    return chain_with_history


def ensure_vectorstore_exists():
    persist_directory = os.path.join(working_dir, "vector_db_dir")
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        st.warning("Base de datos vectorial no encontrada. Reconstruyendo...")
        # Importa y ejecuta tu script de vectorización
        import vectorize_documents
        vectorize_documents.main()
    return startup_vectorstore()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []
    
try:
    if "vectorstore" not in st.session_state:
        with st.spinner("Inicializando base de datos vectorial..."):
            st.session_state.vectorstore = ensure_vectorstore_exists()
            
    if "conversational_chain" not in st.session_state:
        with st.spinner("Configurando el modelo de lenguaje..."):
            st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)
except Exception as e:
    st.error(f"Error al inicializar: {str(e)}")
    st.info("Asegúrate de que la base de datos vectorial esté disponible o ejecuta primero el script de vectorización.")
    st.stop()

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
            response = st.session_state.conversational_chain.invoke(
                {"question": user_input},
                {"configurable": {"session_id": "default"}}
            )
            assistant_message = response["answer"]
            
            # Store source documents for reference
            source_docs = response.get("source_documents", [])
            sources = [doc.metadata.get("source", doc.metadata.get("filename", "desconocido")) 
                      for doc in source_docs]
            
            
            st.markdown(assistant_message)
            st.session_state.messages.append({"role": "assistant", "content": assistant_message})

