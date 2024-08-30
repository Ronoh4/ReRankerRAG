# Import modules and classes
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain_core.documents import Document as LangDocument
from llama_index.core import Document as LlamaDocument
from llama_index.core import Settings
from llama_parse import LlamaParse
import os

# Set environmental variables
os.environ["NVIDIA_API_KEY"] = "YOUR_NVIDIA_KEY"
nvidia_api_key = os.environ["NVIDIA_API_KEY"]

os.environ["LLAMAPARSE_API_KEY"] = "YOUR_LLAMAPARSE_KEY"
llamaparse_api_key = os.environ["LLAMAPARSE_API_KEY"]

# Initialize ChatNVIDIA, NVIDIARerank, and NVIDIAEmbeddings
client = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    api_key=nvidia_api_key,
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024
)

embed_model = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5", 
    api_key=nvidia_api_key, 
    truncate="NONE"
)

reranker = NVIDIARerank(
  model="nvidia/nv-rerankqa-mistral-4b-v3", 
  api_key=nvidia_api_key,
)

# Set the NVIDIA models globally
Settings.embed_model = embed_model
Settings.llm = client

# Parse the local PDF document
parser = LlamaParse(
    api_key=llamaparse_api_key,
    result_type="markdown",
    verbose=True
)

documents = parser.load_data("C:\\Users\\user\\Documents\\Jan 2024\\Projects\\RAGs\\Files\\PhilDataset.pdf")
print("Document Parsed")

# Split parsed text into chunks for embedding model
def split_text(text, max_tokens=512):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word)
        if current_length + word_length + 1 > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length + 1
        else:
            current_chunk.append(word)
            current_length += word_length + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Generate embeddings for document chunks
all_embeddings = []
all_documents = []

for doc in documents:
    text_chunks = split_text(doc.text)
    for chunk in text_chunks:
        embedding = embed_model.embed_query(chunk)
        all_embeddings.append(embedding)
        all_documents.append(LlamaDocument(text=chunk))
print("Embeddings generated")

# Create and persist index with NVIDIAEmbeddings
index = VectorStoreIndex.from_documents(all_documents, embeddings=all_embeddings, embed_model=embed_model)
index.set_index_id("vector_index")
index.storage_context.persist("./storage")
print("Index created")

# Load index from storage
storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(storage_context, index_id="vector_index")
print("Index loaded")

# Query the index and use output as LLM context
def query_model_with_context(question):

    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(question)

    for node in nodes:
        print(node)

    # Rerank the nodes
    ranked_documents = reranker.compress_documents(
        query=question,
        documents = [LangDocument(page_content=node.text) for node in nodes]
    )

    # Print the most relevant and least relevant node
    print(f"Most relevant node: {ranked_documents[0].page_content}")

    # Use the most relevant node as context
    context = ranked_documents[0].page_content

    # Send context and question to the client (NVIDIA Llama 3.1 8B model)
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": str(question)}
    ]
    completion = client.stream(messages)
    
    # Process response
    response_text = ""
    for chunk in completion:
        if chunk.content is not None:
            response_text += chunk.content
    return response_text

# Test with a question 
question = "Which four subjects did philemon handled in academic writing?"
response = query_model_with_context(question)
print("Final Respone: ", response)

