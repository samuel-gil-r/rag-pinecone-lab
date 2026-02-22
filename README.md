# LangChain RAG with Pinecone - Lab 2

This project is the second laboratory in the RAG series. The goal is to build a complete
Retrieval-Augmented Generation system that can answer questions based on documents
stored in a vector database.

Instead of relying only on what the model knows from training, the system first
retrieves relevant information from our own documents and then uses that context
to generate a more accurate and grounded answer.

---

## What is RAG?

RAG (Retrieval-Augmented Generation) is an architecture that combines two things:

- **Retrieval**: searching for relevant information in a knowledge base
- **Generation**: using a language model to generate an answer based on that information

This reduces hallucinations because the model answers based on real documents,
not just its training data.

---

## How the system works
```
Documents (.txt files in /data)
          ↓
    Text chunking
          ↓
  Pinecone stores the chunks as vector records
          ↓
  User asks a question
          ↓
  Pinecone searches for the most relevant chunks
          ↓
  Chunks + Question → Prompt → Groq LLM → Final answer
```

This flow is the complete RAG pipeline:
**Retrieve → Augment → Generate**

---

## Project structure
```
langchain-rag-pinecone/
├── data/
│   ├── doc1.txt          # Knowledge base documents
│   ├── doc2.txt
│   └── doc3.txt
├── src/
│   ├── main.py           # Main script: indexes documents and answers questions
│   ├── ingest.py         # Document loading and chunking logic
│   ├── rag.py            # RAG chain construction
│   ├── query.py          # Interactive query mode
│   ├── test_groq.py      # Groq connection test
│   └── test_pinecone.py  # Pinecone connection test
├── .env                  # API keys (not uploaded to GitHub)
├── .env.example          # Example .env without real keys
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.9 or higher
- Free account at [Groq](https://console.groq.com) for the LLM
- Free account at [Pinecone](https://www.pinecone.io) for the vector database

---

## Installation and setup

**1. Clone the repository:**
```bash
git clone https://github.com/tu-usuario/langchain-rag-pinecone.git
cd langchain-rag-pinecone
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv

# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up Pinecone:**

- Go to [pinecone.io](https://www.pinecone.io) and create a free account
- Go to **Indexes** → **Create Index**
- Select an **integrated embedding model** (llama-text-embed-v2)
- Set metric to **cosine**
- Copy your index name and API key

**5. Create your `.env` file** in the root of the project:
```
GROQ_API_KEY=your-groq-key-here
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_INDEX_NAME=your-index-name-here
```

**6. Add your documents** as `.txt` files inside the `data/` folder.

---

## How to run it
```bash
python src/main.py
```

This will:
1. Read all `.txt` files from the `data/` folder
2. Split them into chunks and upload them to Pinecone
3. Run a set of test questions and print the answers

---

## Output example
```
====================================
QUESTION:
What does the document say about RAG?

=== RETRIEVED CONTEXT ===
RAG combines information retrieval with text generation using language models.

The Hydra-21 project was developed internally by the company TechNova.

Hydra-21 optimizes logistics processes using hybrid models.

=== FINAL ANSWER ===
The document mentions that RAG combines information retrieval with text generation using language models.

====================================
QUESTION:
What role does Pinecone play in this system?

=== RETRIEVED CONTEXT ===
Pinecone is a vector database used for semantic search.

Embeddings convert text into numerical vectors to measure semantic similarity.

=== FINAL ANSWER ===
Pinecone acts as a vector database, storing embeddings and retrieving the most relevant fragments based on the user's query.

===================================
====================================
```
<img width="801" height="574" alt="image" src="https://github.com/user-attachments/assets/3b4f5e0f-8092-4a25-856b-8a72c6d3910b" />

<img width="1600" height="1008" alt="image" src="https://github.com/user-attachments/assets/a14cd412-9d25-43ac-a4da-0f56e790649d" />

---

## Technologies used

- [LangChain](https://python.langchain.com/) — framework for orchestrating prompts, models and parsers
- [Pinecone](https://www.pinecone.io/) — vector database for semantic search
- [Groq](https://console.groq.com/) — free LLM provider
- [llama-3.3-70b-versatile](https://console.groq.com/docs/models) — language model used
- [python-dotenv](https://pypi.org/project/python-dotenv/) — environment variable management

---

## Why Groq instead of OpenAI?

Groq offers free access to powerful open source models like `llama-3.3-70b-versatile`.
Since LangChain is compatible with both providers, switching from OpenAI to Groq only
requires changing the model class in the code. The rest of the architecture stays exactly
the same.

---

## References

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Pinecone LangChain Integration](https://python.langchain.com/docs/integrations/vectorstores/pinecone)
- [Groq Models](https://console.groq.com/docs/models)

---

## Author
- Samuel Antonio Gil Romero

## Course
- TDSE-Transformacion Digital y Soluciones Empresariales 
