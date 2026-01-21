import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Securely set your Cerebras AI API key using Colab's secrets management
# This part assumes the key is already set in the Colab notebook environment previously.
# For script execution, it's typically read from environment variables or passed.
# Re-initialize the LLM, prompt, and chain exactly as defined in cell cuLeLIs3BhiS for consistency.

# IMPORTANT: Make sure os.environ["CEREBRAS_AI_KEY"] is set in the main notebook environment
# before running this script if you execute it via subprocess.
# For direct execution within Colab, userdata.get() is fine, but subprocess needs os.environ.
# We will pass the env variable explicitly in the calling cell.

llm = ChatOpenAI(
    model="gpt-oss-120b", # Consistent with cuLeLIs3BhiS
    base_url="https://api.cerebras.ai/v1",
    api_key=os.environ.get("CEREBRAS_AI_KEY") # Use .get() for safety in scripts
)

prompt = PromptTemplate(
    input_variables=["context"],
    template="""Your task is to generate one factual question and its ground truth answer STRICTLY from the provided context. Do NOT hallucinate. Ensure the question and answer are verbose, detailed, and directly contextual to the provided text. Ignore any non-factual filler like '----------------------------------'. Return the output as a JSON object with the following keys: 'question', 'contexts', 'answer', and 'ground_truth'.
- 'question' should be a well-formulated, verbose question based on the context.
- 'contexts' should be a list containing the provided context.
- 'answer' should be a comprehensive, detailed ground truth answer, directly extracted or summarized from the context.
- 'ground_truth' should be the same as 'answer', representing the comprehensive ground truth.

Context:
{context}

Return JSON:

{{
  "question": "...",
  "contexts": [...],
  "answer": "...",
  "ground_truth": "..."
}}"""
)

chain = prompt | llm | JsonOutputParser()

# Load the processed_chunks.json file
processed_chunks_file = "processed_chunks.json"
if not os.path.exists(processed_chunks_file):
    print(f"Error: {processed_chunks_file} not found. Please run the chunk processing script first.")
    exit()

with open(processed_chunks_file, "r") as f:
    processed_chunks = json.load(f)

# Create an empty list to store the generated Q&A dataset
qna_dataset = []

# Iterate through the loaded chunks, grouping them into batches of 5
batch_size = 5
max_batches = 100 # Limit to 100 batches to generate 100 Q&A pairs (100 * 5 = 500 chunks)

for i in range(0, min(len(processed_chunks), max_batches * batch_size), batch_size):
    batch = processed_chunks[i : i + batch_size]

    # Combine the content of the chunks into a single context string
    combined_context_content = "\n---\n".join([chunk["content"] for chunk in batch])
    batch_chunk_ids = [chunk["id"] for chunk in batch]

    try:
        # Invoke the chain to generate a question-answer pair
        qa_pair = chain.invoke({"context": combined_context_content})

        # From the LLM's JSON output, extract the 'question' and 'answer'
        # and construct a new dictionary for each generated Q&A pair.
        qa_entry = {
            "question": qa_pair.get("question"),
            "answer": qa_pair.get("answer"),
            "source_chunk_ids": batch_chunk_ids, # List of IDs from the 5 source chunks
            "contexts": qa_pair.get("contexts"),
            "ground_truth": qa_pair.get("ground_truth")
        }
        qna_dataset.append(qa_entry)
        print(f"Successfully generated Q&A for batch {i // batch_size + 1}/{max_batches}")
    except Exception as e:
        print(f"Error generating Q&A for batch {i // batch_size + 1}: {e}")

# Save the accumulated Q&A dataset list to a new JSON file
output_qna_file = "ragas_qa_dataset.json"
with open(output_qna_file, "w") as f:
    json.dump(qna_dataset, f, indent=2)

print(f"Generated {len(qna_dataset)} Q&A pairs and saved to {output_qna_file}")
