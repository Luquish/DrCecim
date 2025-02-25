# vectorize_documents.py
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings

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
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Documentos divididos en {len(chunks)} fragmentos.")

# Create Chroma database
print("Creando base de datos Chroma...")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vector_db_dir"
)
print("Base de datos Chroma creada y datos persistidos.")

print("Documentos vectorizados.")
