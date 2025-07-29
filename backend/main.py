# from sqltool import create_agent
# from langchain_community.utilities import SQLDatabase
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# import os
# import sys
# import tiktoken
# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# from uuid import uuid4
# from db import SessionLocal, ChatSession, Conversation
# import uvicorn
# from langgraph.checkpoint.memory import InMemorySaver
# from langgraph.graph import StateGraph, MessagesState, START, END
# from langchain_core.messages import AIMessage

# # ===============================
# # Load environment variables
# # ===============================
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
# if not groq_api_key:
#     print("Error: GROQ_API_KEY environment variable not set.")
#     sys.exit(1)

# # ===============================
# # Initialize LLM
# # ===============================
# llm = ChatOpenAI(
#     model_name="llama3-70b-8192",
#     openai_api_base="https://api.groq.com/openai/v1",
#     openai_api_key=groq_api_key,
#     temperature=0
# )

# # ===============================
# # Connect to SQLite DB
# # ===============================
# try:
#     sql_db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# except Exception as e:
#     print(f"Error connecting to database: {e}")
#     sys.exit(1)

# # ===============================
# # Create Hybrid Agent (SQL + RAG)
# # ===============================
# agent = create_agent(sql_db, llm)

# # ===============================
# # Tokenizer for token counting
# # ===============================
# tokenizer = tiktoken.get_encoding("cl100k_base")

# # ===============================
# # Memory Setup
# # ===============================
# memory = InMemorySaver()

# # ===============================
# # Define workflow (node-based)
# # ===============================
# workflow = StateGraph(state_schema=MessagesState)

# def call_sql_rag(state: MessagesState):
#     """Node that calls the SQL+RAG agent using your main system prompt."""
#     user_query = state["messages"][-1].content

#     result = None
#     for step in agent.stream(
#         {"messages": [{"role": "user", "content": user_query}]},
#         stream_mode="values"
#     ):
#         if step.get("messages"):
#             final_msg = step["messages"][-1]
#             result = getattr(final_msg, "content", str(final_msg))

#     return {"messages": [AIMessage(content=result)]}

# workflow.add_node("sql_rag", call_sql_rag)
# workflow.add_edge(START, "sql_rag")
# workflow.add_edge("sql_rag", END)

# graph = workflow.compile(checkpointer=memory)

# # ===============================
# # FastAPI Setup
# # ===============================
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class QueryRequest(BaseModel):
#     query: str
#     session_id: str = None

# @app.get("/")
# def read():
#     return "it runs"

# @app.post("/api/ask")
# async def ask_question(req: QueryRequest):
#     session_id = req.session_id or str(uuid4())
#     db_session = SessionLocal()

#     try:
#         # Ensure chat session exists
#         chat_session = db_session.query(ChatSession).filter_by(id=session_id).first()
#         if not chat_session:
#             chat_session = ChatSession(id=session_id, title="untitled", total_tokens=0)
#             db_session.add(chat_session)
#             db_session.commit()

#         # Call the graph with memory
#         result_state = graph.invoke(
#             {"messages": [{"role": "user", "content": req.query}]},
#             {"configurable": {"thread_id": str(session_id)}}
#         )

#         # Get final agent reply
#         messages = result_state.get("messages", [])
#         result = messages[-1].content if messages else None

#         # Token usage
#         query_tokens = len(tokenizer.encode(req.query))
#         response_tokens = len(tokenizer.encode(result)) if result else 0
#         total_tokens = query_tokens + response_tokens

#         # Save conversation in DB
#         convo = Conversation(
#             session_id=session_id,
#             user_message=req.query,
#             bot_response=result,
#             tokens_used=total_tokens
#         )
#         db_session.add(convo)
#         chat_session.total_tokens += total_tokens
#         db_session.commit()

#         print(f"\n📊 Token Usage for Query: \"{req.query}\"")
#         print(f"    🔷 Query tokens:    {query_tokens}")
#         print(f"    🔷 Response tokens: {response_tokens}")
#         print(f"    🔷 TOTAL tokens:    {total_tokens}\n")

#         return {"answer": result, "session_id": session_id}

#     except Exception as e:
#         return {"error": str(e)}

#     finally:
#         db_session.close()

# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)


from sqltool import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("Error: GROQ_API_KEY environment variable not set.")
    sys.exit(1)

# Initialize LLM
llm = ChatOpenAI(
    model_name="llama3-70b-8192",
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=groq_api_key,
    temperature=0
)

# Connect to SQLite DB
try:
    sql_db = SQLDatabase.from_uri("sqlite:///Chinook.db")
except Exception as e:
    print(f"Error connecting to database: {e}")
    sys.exit(1)

# Create SQL + RAG Agent
agent = create_agent(sql_db, llm)
print("✅ SQL+RAG Agent created successfully.")
print("👉 Type your question (or 'exit' to quit)")

# Terminal loop
while True:
    question = input("\n📝 Question: ").strip()
    if question.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break

    print("\n🤖 Answering...\n")
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
