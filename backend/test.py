# test_agent_routing.py
import os
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from sqltool import create_agent

# ====== Setup ======
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatOpenAI(
    model_name="llama3-70b-8192",
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# Create hybrid agent
agent = create_agent(db, llm)

# ====== Test queries ======
test_queries = [
    ("SQL test", "List the top 5 albums by sales"),     # SQL path
    ("RAG test", "Who is the CEO of JBS?"),            # RAG path
    ("SQL schema", "Show me all the tables"),          # SQL path
    ("RAG services", "What services does JBS provide?") # RAG path
]

# ====== Run tests ======
for label, query in test_queries:
    print("\n" + "="*80)
    print(f"🔍 TEST: {label}")
    print(f"📝 Query: {query}")
    print("="*80)

    response = agent.invoke({"input": query})

    print("\n💡 Agent response:")
    print(response)

print("\n✅ Test complete. Watch console logs for '[DEBUG] RAG tool called...' to confirm RAG path.")
