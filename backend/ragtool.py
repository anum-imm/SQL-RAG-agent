from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("❌ Please set GROQ_API_KEY in your .env file")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(
    "faiss_index",  # adjust path if needed
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = ChatOpenAI(
    model_name="llama3-70b-8192",
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=groq_api_key,
    temperature=0
)

def rag_tool_fn(query: str) -> str:
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n".join(doc.page_content for doc in retrieved_docs)

    system_prompt = (
        "You are a helpful RAG assistant. Use only the retrieved context from the document to answer.\n\n"
        "If the user query is out of context DO NOT answer it.\n"
        f"Context:\n{context}"
    )
    messages = [SystemMessage(content=system_prompt), {"role": "user", "content": query}]
    response = llm.invoke(messages)
    return response.content

rag_tool = Tool(
    name="Document QA (RAG)",
    func=rag_tool_fn,
    description="Use this tool to answer document-based questions related to jbs. Ideal for questions NOT related to SQL or database schema.."
)
