# vectorize_documents.py
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil
import os
import datetime
import hashlib

# Initialize embeddings
print("Inicializando embeddings...")
# Using a better model for domain-specific knowledge
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embeddings inicializados.")

# Load documents
print("Cargando documentos desde el directorio 'data'...")
loader = DirectoryLoader(path="data",
                        glob="**/*.pdf",  # Include subdirectories
                        loader_cls=UnstructuredFileLoader,
                        show_progress=True)

documents = loader.load()
print(f"Documentos cargados: {len(documents)} documentos encontrados.")

# Extract metadata from filenames and content
def extract_metadata(doc, idx):
    # Generate a unique ID for the document
    doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()[:10]
    
    # Extract filename from source if available
    filename = "unknown"
    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
        filename = os.path.basename(doc.metadata['source'])
    
    # Create enriched metadata
    metadata = {
        "doc_id": doc_id,
        "chunk_id": idx,
        "filename": filename,
        "source": filename,
        "chunk_type": "text",
        "created_at": datetime.datetime.now().isoformat(),
    }
    
    # Preserve any existing metadata
    if hasattr(doc, 'metadata'):
        metadata.update(doc.metadata)
        
    return metadata

# Enrich documents with metadata
for i, doc in enumerate(documents):
    doc.metadata.update(extract_metadata(doc, i))

# Split documents into chunks using recursive splitter for better semantic chunking
print("Dividiendo documentos en fragmentos...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,          # Smaller chunks for more precise retrieval
    chunk_overlap=100,       # Larger overlap to maintain context between chunks
    separators=["\n\n", "\n", ". ", " ", ""],  # Try to split on paragraphs first
    length_function=len,
)

chunks = text_splitter.split_documents(documents)
print(f"Documentos divididos en {len(chunks)} fragmentos.")

# Update metadata for each chunk
for i, chunk in enumerate(chunks):
    # Preserve original document metadata but update chunk-specific info
    chunk.metadata.update({
        "chunk_id": i,
        "total_chunks": len(chunks)
    })

# Create Chroma database
print("Creando base de datos Chroma...")

# Create backup of existing database if it exists
if os.path.exists("vector_db_dir"):
    backup_dir = f"vector_db_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Creando backup de la base de datos existente en {backup_dir}...")
    shutil.copytree("vector_db_dir", backup_dir)
    shutil.rmtree("vector_db_dir")

# Create new Chroma instance with specific configuration
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vector_db_dir",
    collection_metadata={"hnsw:space": "cosine"}  # Using cosine similarity
)

print("Base de datos Chroma creada y datos persistidos.")

# Print some statistics
print(f"Estadísticas de vectorización:")
print(f"- Documentos procesados: {len(documents)}")
print(f"- Fragmentos creados: {len(chunks)}")
print(f"- Tamaño promedio de fragmento: {sum(len(c.page_content) for c in chunks) / len(chunks):.2f} caracteres")
print(f"- Base de datos guardada en: {os.path.abspath('vector_db_dir')}")

print("Documentos vectorizados exitosamente.")