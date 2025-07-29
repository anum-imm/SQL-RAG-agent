from sqltool import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import sys
import tiktoken
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from db import SessionLocal, ChatSession, Conversation  

# ===============================
# Load environment variables
# ===============================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("Error: GROQ_API_KEY environment variable not set.")
    sys.exit(1)

# ===============================
# Initialize LLM
# ===============================
llm = ChatOpenAI(
    model_name="llama3-70b-8192",
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=groq_api_key,
    temperature=0
)

# ===============================
# Connect to SQLite DB
# ===============================
try:
    sql_db = SQLDatabase.from_uri("sqlite:///Chinook.db")
except Exception as e:
    print(f"Error connecting to database: {e}")
    sys.exit(1)

# ===============================
# Create Hybrid Agent (SQL + RAG)
# ===============================
agent = create_agent(sql_db, llm)

# ===============================
# Tokenizer for token counting
# ===============================
tokenizer = tiktoken.get_encoding("cl100k_base")

# ===============================
# FastAPI Setup
# ===============================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    session_id: str = None

@app.get("/")
def read():
    return "it runs"

@app.post("/api/ask")
async def ask_question(req: QueryRequest):
    session_id = req.session_id or str(uuid4())
    db_session = SessionLocal()

    try:
        # Ensure chat session exists
        chat_session = db_session.query(ChatSession).filter_by(id=session_id).first()
        if not chat_session:
            chat_session = ChatSession(id=session_id, title="untitled", total_tokens=0)
            db_session.add(chat_session)
            db_session.commit()

        # Call the SQL agent
        result = None
        for step in agent.stream(
            {"messages": [{"role": "user", "content": req.query}]},
            stream_mode="values"
        ):
            if step.get("messages"):
                final_msg = step["messages"][-1]
                result = getattr(final_msg, "content", str(final_msg))

        # Token usage tracking
        query_tokens = len(tokenizer.encode(req.query))
        response_tokens = len(tokenizer.encode(result)) if result else 0
        total_tokens = query_tokens + response_tokens

        # Save conversation
        convo = Conversation(
            session_id=session_id,
            user_message=req.query,
            bot_response=result,
            tokens_used=total_tokens
        )
        db_session.add(convo)

        # Update total tokens for the session
        chat_session.total_tokens += total_tokens
        db_session.commit()

        # Debug token usage
        print(f"\n📊 Token Usage for Query: \"{req.query}\"")
        print(f"    🔷 Query tokens:    {query_tokens}")
        print(f"    🔷 Response tokens: {response_tokens}")
        print(f"    🔷 TOTAL tokens:    {total_tokens}\n")

        return {
            "answer": result,
            "session_id": session_id
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        db_session.close()
