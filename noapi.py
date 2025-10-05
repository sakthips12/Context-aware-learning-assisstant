from typing import List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

# LangChain community
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Local models
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# 1. State
# ----------------------
class TutorState(TypedDict, total=False):
    query: str
    mode: str
    docs: List[str]
    output: str
    error: Optional[str]

# ----------------------
# 2. Setup Local LLM + Embeddings
# ----------------------
try:
    llm = ChatOllama(model="llama3")  # Local Ollama model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    logger.error(f"Failed to initialize models: {e}")
    raise

# ----------------------
# 3. Build VectorStore
# ----------------------
def build_vectorstore(pdf_path: str):
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")
        raise

try:
    vectorstore = build_vectorstore("lecture.pdf")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    # You might want to handle this more gracefully in production

# ----------------------
# 4. Node Functions
# ----------------------
def router_node(state: TutorState) -> TutorState:
    try:
        query = state["query"].lower()
        state["mode"] = "quiz" if "quiz" in query or "mcq" in query else "explanation"
        return state
    except Exception as e:
        state["error"] = f"Error in router: {str(e)}"
        return state

def retrieve_node(state: TutorState) -> TutorState:
    try:
        docs = retriever.invoke(state["query"])
        state["docs"] = [d.page_content for d in docs]
        return state
    except Exception as e:
        state["error"] = f"Error in retrieval: {str(e)}"
        return state

def explanation_node(state: TutorState) -> TutorState:
    try:
        docs_text = " ".join(state["docs"])
        response = llm.invoke(f"Explain this clearly for a student:\n\n{docs_text}")
        state["output"] = response.content
        return state
    except Exception as e:
        state["error"] = f"Error in explanation: {str(e)}"
        return state

def quiz_node(state: TutorState) -> TutorState:
    try:
        docs_text = " ".join(state["docs"])
        prompt = f"""
        Based on the text below, create 3 multiple-choice questions (MCQs).
        Each should have 4 options (A-D) with one correct answer.
        Return in JSON format with the following structure:
        {{
          "questions": [
            {{
              "question": "Question text",
              "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
              "correct_answer": "A"
            }}
          ]
        }}
        
        Text: {docs_text}
        """
        response = llm.invoke(prompt)
        
        # Try to parse as JSON, fallback to raw text if it fails
        try:
            quiz_data = json.loads(response.content)
            state["output"] = json.dumps(quiz_data, indent=2)
        except json.JSONDecodeError:
            state["output"] = response.content
            
        return state
    except Exception as e:
        state["error"] = f"Error in quiz generation: {str(e)}"
        return state

# ----------------------
# 5. Graph
# ----------------------
graph = StateGraph(TutorState)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("explain", explanation_node)
graph.add_node("quiz", quiz_node)
graph.set_entry_point("router")
graph.add_edge("router", "retrieve")
graph.add_conditional_edges("retrieve", lambda s: s["mode"], {"explanation": "explain", "quiz": "quiz"})
graph.add_edge("explain", END)
graph.add_edge("quiz", END)
app = graph.compile()

if __name__ == "__main__":
    print("\n--- Explanation ---\n")
    result = app.invoke({"query": "Explain backpropagation"})
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(result["output"])
        
    print("\n--- Quiz ---\n")
    result = app.invoke({"query": "Generate a quiz on CNNs"})
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(result["output"])