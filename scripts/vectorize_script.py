import hashlib
import json
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Define the absolute path to the PDF file
pdf_file_path = "/content/BOAT CREW HANDBOOK - 16114.1B_Boat Operations.pdf"

# 1. Load the PDF file
if not os.path.exists(pdf_file_path):
    print(f"Error: PDF file not found at {pdf_file_path}")
    exit()

loader = PyPDFLoader(pdf_file_path)
docs = loader.load()

# 2. Initialize RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=850, # Updated to 850 for consistency
    chunk_overlap=120 # Updated to 120 for consistency
)

# 3. Split the loaded PDF documents into chunks
chunks = splitter.split_documents(docs)

# 4. Create an empty list to store the processed chunks
processed_chunks_list = []

# 5. Iterate through each chunk, generate hash ID, and add metadata
for i, chunk in enumerate(chunks):
    # Generate a unique hash ID for the chunk.page_content
    hash_id = hashlib.md5(chunk.page_content.encode('utf-8')).hexdigest()

    # Create a dictionary for each processed chunk
    processed_chunk = {
        "content": chunk.page_content,
        "id": hash_id,
        "metadata": chunk.metadata
    }

    # Append this dictionary to your list of processed chunks
    processed_chunks_list.append(processed_chunk)

# 6. Save the list of processed chunks to a JSON file
output_filename = "processed_chunks.json"
with open(output_filename, "w") as f:
    json.dump(processed_chunks_list, f, indent=2)

print(f"Processed {len(processed_chunks_list)} chunks and saved to {output_filename}")
