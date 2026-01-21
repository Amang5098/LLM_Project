import json
import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 1. Define the path to the processed chunks file
processed_chunks_file = "processed_chunks.json"

# 2. Check if the processed_chunks.json file exists
if not os.path.exists(processed_chunks_file):
    print(f"Error: {processed_chunks_file} not found. Please ensure the chunk processing script was run successfully.")
    exit()

# 3. Load the processed chunks
with open(processed_chunks_file, "r") as f:
    processed_chunks_data = json.load(f)

# 4. Initialize an embedding model
# Using 'BAAI/bge-small-en-v1.5' for efficiency and good performance
embeddings = HuggingFaceBgeEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# 5. Prepare Document objects for vector store
# Each Document object should contain page_content and metadata
documents_for_vectorstore = []
for chunk in processed_chunks_data:
    # Ensure metadata includes the original chunk's id and any other metadata
    doc_metadata = chunk.get("metadata", {})
    doc_metadata["id"] = chunk.get("id") # Add the hash ID to metadata for retrieval

    doc = Document(
        page_content=chunk.get("content"),
        metadata=doc_metadata
    )
    documents_for_vectorstore.append(doc)

# 6. Create an in-memory FAISS vector store
print("Creating FAISS vector store from documents...")
vectorstore = FAISS.from_documents(documents_for_vectorstore, embeddings)
print(f"Successfully created FAISS vector store with {len(documents_for_vectorstore)} documents.")

# --- Verification Step ---
print("\n--- Verification: Performing a sample retrieval query ---")

# Define a sample query string
query = "What are the steps for pre-underway checks?"

# Perform a similarity search
retrieved_docs = vectorstore.similarity_search(query, k=3) # Retrieve top 3 relevant chunks

print(f"Query: '{query}'")
print(f"Retrieved {len(retrieved_docs)} documents:")

# Iterate through retrieved documents and print their content and hash IDs
for i, doc in enumerate(retrieved_docs):
    print(f"\nDocument {i+1}:")
    print(f"Content (first 200 chars): {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")
    print(f"Hash ID from metadata: {doc.metadata.get('id', 'N/A')}")

print("\nVerification complete.")
