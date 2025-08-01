# rag_tool.py
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

# ===============================
# Load environment variables
# ===============================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY environment variable not set.")

# ===============================
# Load embeddings and FAISS index
# ===============================
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    raise RuntimeError(f"Failed to load HuggingFace embeddings: {e}")

try:
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    raise RuntimeError(f"Failed to load FAISS index: {e}")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ===============================
# LLM for RAG responses
# ===============================
llm = ChatOpenAI(
    model_name="llama3-70b-8192",
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=groq_api_key,
    temperature=0
)

# ===============================
# Function to run RAG retrieval
# ===============================
def rag_tool_fn(query: str) -> str:
    print("\n📌 [DEBUG] RAG tool called with query:", query)

    """
    Retrieve relevant context from FAISS and answer the query.
    If no relevant context is found, return a 'no info' message.
    """
    try:
        retrieved_docs = retriever.invoke(query)
    except Exception as e:
        return f"Error retrieving documents: {e}"

    context = "\n".join(doc.page_content for doc in retrieved_docs)

    if not context.strip():
        return "I couldn't find relevant information in the documents."

    system_prompt = (
        "You are a helpful RAG assistant. "
        "Use only the retrieved context from the documents to answer.\n\n"
        "If the user query is out of context, DO NOT answer it.\n"
        f"Context:\n{context}"
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]

    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error generating response: {e}"

# ===============================
# LangChain Tool definition
# ===============================
rag_tool = Tool(
    name="rag_tool",
    func=rag_tool_fn,
    description="Use this tool to answer document-based questions from FAISS index (RAG)."
)
