# 📌 SQL + RAG Agent with Analytics Tool (5-Tool Intelligent Agent)

An advanced **multi-tool AI agent** capable of intelligently deciding how to answer user queries using a combination of:

1. **SQL Agent** → Runs SQL queries on structured databases.
2. **RAG Agent** → Uses Retrieval-Augmented Generation with FAISS to search unstructured knowledge.
3. **Analytics Tool** → Generates charts and visual insights from SQL query results.
4. **Knowledge QA Tool** → Answers factual/general knowledge questions using the RAG pipeline.
5. **Fallback Conversational Tool** → Provides a natural language fallback when other tools are not suitable.

The system **automatically decides** which tool to use based on the nature of the question.

---

## 🚀 Features

- **Hybrid Intelligence**
  - **SQL Agent** → Structured data queries from `Chinook.db` (or any SQLite DB).
  - **RAG Agent** → Semantic search across unstructured embeddings.
  - **Analytics Tool** → Converts SQL query results into visualizations (bar chart, histogram, line chart).
  - **Knowledge QA Tool** → Short, fact-based answers using embedded content.
  - **Fallback Conversational Tool** → Keeps the conversation flowing for unsupported queries.

- **Automatic Tool Selection** → No need to manually choose; the agent routes the request to the correct tool.
- **Interactive API** → Explore functionality via [Swagger UI](http://localhost:8000/docs).
- **Dockerized Deployment** → Works anywhere using Docker Compose.

---

## 🛠️ Tech Stack

- **Backend** → [FastAPI](https://fastapi.tiangolo.com/)
- **Database** → SQLite (`Chinook.db`)
- **Vector Store** → [FAISS](https://github.com/facebookresearch/faiss)
- **Agent Framework** → [LangChain](https://www.langchain.com/)
- **Visualization** → Matplotlib & Pandas

---

- **Local Installation**

-cd backend
-python -m venv venv
-source venv/bin/activate  # Windows: venv\Scripts\activate
-pip install --upgrade pip
-pip install -r requirements.txt


-**Run locally**

uvicorn main:app --reload --host 0.0.0.0 --port 8000
Visit → http://localhost:8000/docs

