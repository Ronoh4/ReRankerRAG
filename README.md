RAG with NVIDIA Reranker
This project implements a Retrieval-Augmented Generation (RAG) system that leverages NVIDIA's AI models for text generation, embeddings, and reranking. It is designed to improve the quality of responses to user queries by integrating a reranking mechanism that prioritizes the most relevant context. The application processes and stores information from a local PDF document, then uses that stored data to generate accurate and contextually relevant answers to user queries.

How It Works
Document Parsing and Chunking: The application begins by parsing a local PDF document using the LlamaParse API. The parsed text is then split into manageable chunks to be processed by the embedding model.

Embeddings Generation: Each chunk of text is converted into vector embeddings using NVIDIA's Embeddings model. These embeddings represent the semantic content of the text, making it easier to perform similarity searches.

Index Creation and Storage: A vector index is created using the generated embeddings, which allows for efficient retrieval of relevant information. The index is stored locally for persistence across sessions.

Query Processing: When a user query is received, the system retrieves the top relevant text chunks from the index based on similarity. These retrieved chunks form the initial context for generating a response.

Reranking: The retrieved chunks are reranked using NVIDIA's Reranker model to determine the most relevant context. This step helps refine the context used for generating a more accurate and relevant response.

Text Generation: The highest-ranked context is combined with the user query and sent to the NVIDIA Llama 3.1 model for generating a final response. The system streams the response back to the user, ensuring timely and informative answers.

Key Features
Retrieval-Augmented Generation (RAG): Enhances the quality of responses by combining information retrieval with generative capabilities.
NVIDIA AI Models: Uses cutting-edge NVIDIA models for text embeddings, reranking, and generation, providing high performance and accuracy.
Scalable and Persistent Index: The vector index allows for efficient and scalable retrieval of relevant information and is stored persistently for continuous use.
Contextual Reranking: Ensures that the most relevant context is used in generating responses, improving accuracy and relevance.
Usage
Set up API keys for NVIDIA and LlamaParse.
Parse and process the desired PDF document to generate embeddings and create a vector index.
Use the RAG system to query the stored knowledge base and receive contextual, accurate responses to specific questions.
