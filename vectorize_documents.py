# vectorize_documents.py
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil
import os

# Initialize embeddings
print("Inicializando embeddings...")
embeddings = HuggingFaceEmbeddings()
print("Embeddings inicializados.")

# Load documents
print("Cargando documentos desde el directorio 'data'...")
loader = DirectoryLoader(path="data",
                        glob="*.pdf",
                        loader_cls=UnstructuredFileLoader)

documents = loader.load()
print(f"Documentos cargados: {len(documents)} documentos encontrados.")

# Split documents into chunks
print("Dividiendo documentos en fragmentos...")
text_splitter = CharacterTextSplitter(
    chunk_size=500,          # Reducir tamaño para mayor precisión
    chunk_overlap=50,        # Overlap más pequeño
    separator="\n",          # Usar saltos de línea como separador natural
    length_function=len,
)

chunks = text_splitter.split_documents(documents)
for i, chunk in enumerate(chunks):
    chunk.metadata.update({
        "chunk_id": i,
        "source": "medical_info",
        "chunk_type": "text"
    })

print(f"Documentos divididos en {len(chunks)} fragmentos.")

# Create Chroma database
print("Creando base de datos Chroma...")

# Limpiar directorio de la base de datos si existe
if os.path.exists("vector_db_dir"):
    shutil.rmtree("vector_db_dir")

# Crear nueva instancia de Chroma con configuración específica
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vector_db_dir",
)
print("Base de datos Chroma creada y datos persistidos.")

print("Documentos vectorizados.")
