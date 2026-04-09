# renuka.chitmalwar
Capstone projects for renuka.chitmalwar@neuleap.ai

#  Biomedical Research Assistant using RAG (Retrieval-Augmented Generation)

A **production-style biomedical document Question Answering (Q&A) system** built using **Retrieval-Augmented Generation (RAG)**.  
The system allows users to upload biomedical research PDFs and ask natural language questions, returning **grounded answers with source attribution**.

---

##  Features

- Upload biomedical research PDFs
- Intelligent document retrieval using vector similarity search
- Context-aware answers powered by **Google Gemini**
- Transparent **source attribution** (file name + page number)
- Interactive **Streamlit chat interface**
-  Modular, extensible project structure

---

##  System Architecture
User Query
↓
Retriever (Chroma Vector DB)
↓
Relevant Document Chunks
↓
Prompt + Context
↓
Gemini LLM
↓
Answer + Sources


---

##  Core Concepts Used

- Retrieval-Augmented Generation (RAG)
- Vector Embeddings
- Semantic Similarity Search
- Prompt Engineering
- Source Grounding
- Document Chunking Strategies

---

##  Tech Stack

### Large Language Model
- **Google Gemini** (`ChatGoogleGenerativeAI`)

### Embeddings
- `sentence-transformers`
- Model: `all-MiniLM-L6-v2`

### Vector Database
- **ChromaDB**

### Document Processing
- `unstructured`
- `PyPDF`

### Frameworks & Libraries
- `LangChain`
- `Streamlit`
- `python-dotenv`

---
















