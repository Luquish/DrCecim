from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil
import os
import datetime
import hashlib

# Inicializar embeddings
print("Inicializando embeddings...")
# Considera utilizar un modelo especializado en textos legales si está disponible.
embeddings = HuggingFaceEmbeddings(model_name="dccuchile/bert-base-spanish-wwm-cased")
print("Embeddings inicializados.")

# Cargar documentos desde el directorio 'data'
print("Cargando documentos desde el directorio 'data'...")
loader = DirectoryLoader(
    path="data",
    glob="**/*.pdf",  # Incluir subdirectorios
    loader_cls=UnstructuredFileLoader,
    show_progress=True
)

documents = loader.load()
print(f"Documentos cargados: {len(documents)} documentos encontrados.")

# Función para extraer y enriquecer la metadata de cada documento
def extract_metadata(doc, idx):
    # Generar un ID único basado en el contenido (se puede ajustar según el caso)
    doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()[:10]
    
    # Extraer el nombre del archivo si está disponible en la metadata
    filename = "unknown"
    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
        filename = os.path.basename(doc.metadata['source'])
    
    # Crear metadata enriquecida
    metadata = {
        "doc_id": doc_id,
        "chunk_id": idx,
        "filename": filename,
        "source": filename,
        "chunk_type": "text",
        "created_at": datetime.datetime.now().isoformat(),
    }
    
    # Conservar cualquier metadata existente en el documento
    if hasattr(doc, 'metadata'):
        metadata.update(doc.metadata)
        
    return metadata

# Enriquecer cada documento con su metadata correspondiente
for i, doc in enumerate(documents):
    doc.metadata.update(extract_metadata(doc, i))

# Dividir documentos en fragmentos usando un splitter que respete la estructura legal
print("Dividiendo documentos en fragmentos...")
# Se añaden separadores para mantener la integridad de los artículos y párrafos
separators = ["\nArt. ", "\n\n", "\n", ". ", " ", ""]
# Ajustar chunk_size a 600 para conservar secciones completas, modificable según necesidad
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,          # Tamaño del fragmento ajustado para textos legales
    chunk_overlap=100,       # Solapamiento para mantener continuidad entre fragmentos
    separators=separators,
    length_function=len,
)

chunks = text_splitter.split_documents(documents)
print(f"Documentos divididos en {len(chunks)} fragmentos.")

# Actualizar metadata de cada fragmento con información adicional
for i, chunk in enumerate(chunks):
    chunk.metadata.update({
        "chunk_id": i,
        "total_chunks": len(chunks)
    })

# Crear base de datos Chroma para almacenar los embeddings
print("Creando base de datos Chroma...")

# Si ya existe una base de datos, hacer backup y eliminar la anterior
if os.path.exists("vector_db_dir"):
    backup_dir = f"vector_db_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Creando backup de la base de datos existente en {backup_dir}...")
    shutil.copytree("vector_db_dir", backup_dir)
    shutil.rmtree("vector_db_dir")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vector_db_dir",
    collection_metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 128,  # Aumenta para mayor precisión en la construcción del índice
        "hnsw:search_ef": 96,         # Aumenta para búsquedas más precisas
        "hnsw:M": 16                  # Aumenta las conexiones por nodo
    }
)

print("Base de datos Chroma creada y datos persistidos.")

# Imprimir estadísticas de la vectorización
print(f"Estadísticas de vectorización:")
print(f"- Documentos procesados: {len(documents)}")
print(f"- Fragmentos creados: {len(chunks)}")
print(f"- Tamaño promedio de fragmento: {sum(len(c.page_content) for c in chunks) / len(chunks):.2f} caracteres")
print(f"- Base de datos guardada en: {os.path.abspath('vector_db_dir')}")

print("Documentos vectorizados exitosamente.")
