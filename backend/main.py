import os
from dotenv import load_dotenv
import warnings
from sqlalchemy.exc import SAWarning
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
warnings.filterwarnings("ignore", category=SAWarning)

load_dotenv()

postgres_uri = os.getenv("POSTGRES_URI")
groq_api_key = os.getenv("GROQ_API_KEY")


if not postgres_uri or not groq_api_key:
    raise ValueError("❌ POSTGRES_URI and GROQ_API_KEY must be set in the .env file")

db = SQLDatabase.from_uri(postgres_uri)
llm = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key)


from sqltool import create_agent, get_all_tools 

tools = get_all_tools(db, llm)


agent = create_agent(db, llm) 

print("✅ Agent created successfully with SQL + RAG tools.")
print("👉 Starting test query…\n")

while True:
    question = input("\n📝 Enter your question (or type 'exit' to quit): ").strip()
    
    if question.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break

    print(f"\n🤖 Answering: {question}\n")
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
