import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

# 1. SETUP PERSISTENT STORAGE
# This folder will be created on your disk and house your data
DB_PATH = "./arxiv"
client = chromadb.PersistentClient(path=DB_PATH)

# Use a standard embedding model (runs locally on CPU/GPU)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create or load the collection
collection = client.get_or_create_collection(
    name="Arxiv-Database",
    embedding_function=embedding_func
)

# 2. PROCESSING PIPELINE
PDF_DIRECTORY = "./rag_data"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

for filename in os.listdir(PDF_DIRECTORY):
    if filename.endswith(".pdf"):
        print(f"📄 Processing {filename}...")
        loader = PyPDFLoader(os.path.join(PDF_DIRECTORY, filename))

        # Load and split the PDF into chunks
        pages = loader.load_and_split(text_splitter)

        # Prepare data for ChromaDB
        documents = [page.page_content for page in pages]
        metadatas = [{"source": filename, "page": page.metadata.get("page")} for page in pages]
        ids = [f"{filename}_{i}" for i in range(len(pages))]

        # 3. BATCH INSERTION
        # ChromaDB has a limit (usually ~5000) per insert, so we do it in chunks
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            collection.add(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )

print("✅ Vectorization complete. Database saved to disk.")