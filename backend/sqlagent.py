from langchain_community.agent_toolkits import SQLDatabaseToolkit
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


toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

for tool in tools:
    print(f"{tool.name}: {tool.description}\n")


from langgraph.prebuilt import create_react_agent

system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

agent = create_react_agent(
    llm,
    tools,
    prompt=system_prompt,
)


while True:
    # Take question from user
    question = input("\n📝 Enter your question (or type 'exit' to quit): ").strip()
    
    # Exit condition
    if question.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break

    # Run the agent with streaming & pretty print
    print(f"\n🤖 Answering: {question}\n")
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
