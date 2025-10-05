# Context-aware-learning-assisstant

This project implements an AI-powered tutoring assistant that can explain concepts from PDF lecture notes and generate quizzes automatically. It combines retrieval-augmented generation (RAG) with a graph-based workflow for modular query handling.

🚀 Features
Context-Aware Responses: Uses FAISS vector store + HuggingFace embeddings for relevant context retrieval.
Explanations & Quizzes: Routes user queries to either a detailed explanation or an auto-generated multiple-choice quiz (MCQ).
Graph Workflow: Built with LangGraph for structured state management and modular nodes.
Local LLM Integration: Powered by Ollama (LLaMA3) for private, offline inference.
📂 Project Structure
├── noapi.py              # Entry point with LangGraph pipeline
├── lecture.pdf          # Example lecture file for ingestion
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

⚙️ Setup & Installation

Create a virtual environment & install dependencies:

pip install -r requirements.txt


Ensure you have Ollama installed and a model pulled (e.g., llama3):

ollama pull llama3

📚 Tech Stack
LangGraph – Graph-based orchestration
LangChain – RAG framework
Ollama – Local LLM inference
HuggingFace Embeddings
FAISS – Vector store for semantic search


🔮 Future Improvements
Add a web UI for student interaction.
Support more formats (Word, Markdown, HTML).
Enhance quiz generation with difficulty levels.
