from typing import Optional, List
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage


# ===============================
# Agent Creation
# ===============================

def create_agent(db, llm, verbose=False):
    """
    Create the SQL Agent given a SQLDatabase instance and an LLM.
    """
    tools = get_sql_tools(db, llm)

    dialect = db.dialect
    top_k = 5

    system_prompt = f"""
You are a knowledgeable and cautious SQL agent designed to interact with a {dialect} database.

### Your Behavior:
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. 
You can use the following tools to explore the database and answer user questions:
        list_tables_tool to get the names of the tables,
        get_schema_tool to get the schema of the database,
        sql_query_checker_tool to check if the query generated doesnt contain any error and then send it to run_query_tool to retrieve it 
        run_query_tool to execute a SQL query and return the result.


Unless the user specifies a specific number of examples they wish to obtain, return all matching rows.

###  Rules for Queries:
Always CONSULT the schema before generating a query.
Pay attention to column names and their case — {dialect} is case-sensitive, so use exact names from the schema.
If a column/table name in the schema is quoted , always quote it in the query.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
During checking, matching table or column names to the user question, ignore case, ignore plural/singular, and handle common abbreviations or short forms. If unsure, list available tables and pick the closest match.”
If the user explicitly asks for the schema of a specific table, call `get_schema_tool` with the table name, and output the full schema as returned.
Do not attempt to generate or run any further SQL query after outputting the schema.

### Non-SQL or Irrelevant Questions:
If the user query is not SQL-related or IRRELEVANT to the database, respond immediately:
"Sorry, I can only help with SQL queries about this database. Please ask a SQL-related question."
Do not take any further action or call any tools in this case.

### Safety Rules:
Never run DML or destructive statements such as INSERT, UPDATE, DELETE, DROP, ALTER.
Never modify data.
Never make assumptions about schema — always check first.

### On Errors:
If a query fails due to syntax or column errors:
  - Inspect the error message carefully.
  - Re-check the schema and fix the column/table names or syntax.
  - Retry the corrected query once.
  - If still failing, reply:  
    > `"Unable to retrieve data due to repeated query errors. Please check the input question."`
  - If the tool response is empty or says no data was found, clearly tell the user: "No matching data found for your query."
- Otherwise, display the result exactly as returned by the tool.

To start you should ALWAYS look at the tables in the database to see what you can query. Do NOT skip this step.

Use only the returned result to formulate your final answer.

#LAST STEP
Upon answering check what user has asked and what u should answer it, do not given irrelevant details.
You are now ready to answer any SQL-related question about this database.
"""

    agent = create_react_agent(
        llm,
        tools,
        prompt=system_prompt,
    )

    return agent


# ===============================
# SQL Tools
# ===============================



def list_tables_tool(db: SQLDatabase):
    """
    Tool that lists all tables in the database.
    """
    return Tool(
        name="sql_db_list_tables",
        description="List all tables in the SQL database. Input can be any string.",
        func=lambda _: f"Tables: {', '.join(db.get_usable_table_names())}"
    )


def get_schema_tool(db: SQLDatabase):
    """
    Tool: Get schema for specified tables.
    If found → show schema.
    If not found → 'Table not found: …'
    """
    return Tool(
        name="sql_db_schema",
        description="Get schema for specified tables. Input: comma-separated table names.",
        func=lambda table_names: (
            "No tables specified." if not table_names.strip() else
            "\n\n".join(
                db.get_table_info([name.strip()]) or f"Table not found: {name.strip()}"
                for name in table_names.split(",") if name.strip()
            )
        )
    )

check_query_system_prompt = """
You are a SQL expert with a strong attention to detail.

Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
"""


def sql_query_checker_tool(llm, db):
    prompt = check_query_system_prompt.format(dialect=db.dialect)
    return Tool(
        name="sql_db_query_checker",
        description="Check and fix a SQL query if needed. Input: SQL query as string.",
        func=lambda query: llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]).content
    )


def run_query_tool(db: SQLDatabase):
    """
    Tool that executes a SQL query on the database.
    """
    return Tool(
        name="sql_db_query",
        description="Run a SQL query on the database. Input should be a SQL string.",
        func=lambda query: db.run(query) if query.strip() else "No SQL query provided."
    )


def get_sql_tools(db: SQLDatabase, llm):
    """
    Return all custom SQL tools as a list.
    """
    tools = [
        list_tables_tool(db),
        get_schema_tool(db),
        sql_query_checker_tool(llm, db),
        run_query_tool(db),
    ]

    print("✅ Custom SQL Tools available:")
    for t in tools:
        print(f"- {t.name}: {t.description}")

    return tools
