from sqltool import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import os
import sys
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

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
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
except Exception as e:
    print(f"Error connecting to database: {e}")
    sys.exit(1)

# ===============================
# Create Hybrid Agent (SQL + RAG)
# ===============================
agent = create_agent(db, llm)

print("✅ SQL Agent created successfully.")
print("👉 Starting test query…\n")

while True:
    question = input("\n📝 Enter your question (or type 'exit' to quit): ").strip()
    if question.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break
    print(f"\n🤖 Answering: {question}\n")
    final_answer = None
    # Stream all steps, but only keep the last message
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        if step.get("messages"):
            final_answer = step["messages"][-1]
    # Print only the final answer
    if final_answer is not None:
        # Try to print the content if it's a message object
        content = getattr(final_answer, 'content', str(final_answer))
        print(f"Agent: {content}\n")
    else:
        print("Agent: Sorry, I couldn't generate an answer.\n")
