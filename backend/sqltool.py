from typing import Optional, List
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage
from ragtool import rag_tool

# ===============================
# Agent Creation
# ===============================

def create_agent(db: SQLDatabase, llm):
    """
    Create the hybrid agent with SQL and RAG tools.
    """
    tools = get_all_tools(db, llm)
    dialect = db.dialect

    system_prompt = f"""
You are an intelligent assistant capable of answering both SQL-related questions and document-based (RAG) questions.

## Capabilities

You have access to the following tools:
1. sql_db_list_tables - List all tables in the SQL database.
2. sql_db_schema - Get the schema of one or more tables.
3. sql_db_query_checker - Check and fix SQL queries before running them.
4. sql_db_query - Execute SQL queries.
5. rag_tool - Retrieve information from faiss (RAG-based QA).

## SQL Questions

For questions related to the database:
- Always consult the schema before generating a SQL query.
- Pay attention to column/table names, which may be case-sensitive.
- If a query fails, inspect the error and retry with corrections.
- If still unsuccessful, return:  
  > "Unable to retrieve data due to repeated query errors. Please check the input question."
- If no matching data is found, respond clearly:  
  > "No matching data found for your query."
- NEVER use destructive operations like INSERT, DELETE, UPDATE, DROP, or ALTER.
- NEVER assume the schema — always confirm using schema tools.

Before answering, double-check:
- That the query syntax is correct.
- That the correct tables/columns are used.
- That your response matches what the user asked — do not add unrelated details.

## Document-Based (RAG) Questions

If the user asks a question that is related to jbs and NOT related to SQL or the database schema (e.g., about a topic or concept from uploaded documents):
- Use the `rag_tool` to retrieve the answer.
- Respond concisely and clearly with the relevant information from the documents.
- If no relevant documents are found, say:  
  > "I couldn't find relevant information in the documents."

## Question Classification

Before answering, decide whether the user's question:
- Is a SQL/database-related query → use SQL tools.
- Is a general knowledge or context-based question → use rag_tool.
- Is not answerable using either → respond:  
  > "Sorry, I can only answer SQL questions about this database or questions based on the uploaded documents."

## Final Rule

Do NOT fabricate answers. Only use results returned by tools. Be safe, accurate, and clear in your responses.
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
    def _list_tables(_):
        try:
            tables = db.get_usable_table_names()
            return f"Tables: {', '.join(tables)}" if tables else "No tables found in the database."
        except Exception as e:
            return f"Error listing tables: {e}"

    return Tool(
        name="sql_db_list_tables",
        description="List all tables in the SQL database. Input can be any string.",
        func=_list_tables
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
    """
    Tool to check and fix SQL queries using the LLM.
    """
    prompt = check_query_system_prompt.format(dialect=db.dialect)
    def _check_query(query):
        try:
            result = llm.invoke([
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ])
            return result.content
        except Exception as e:
            return f"Error checking SQL query: {e}"

    return Tool(
        name="sql_db_query_checker",
        description="Check and fix a SQL query if needed. Input: SQL query as string.",
        func=_check_query
    )

def run_query_tool(db: SQLDatabase):
    """
    Tool that executes a SQL query on the database.
    """
    def _run_query(query):
        try:
            if not query.strip():
                return "No SQL query provided."
            result = db.run(query)
            if not result:
                return "No matching data found for your query."
            return result
        except Exception as e:
            return f"Error running SQL query: {e}"

    return Tool(
        name="sql_db_query",
        description="Run a SQL query on the database. Input should be a SQL string.",
        func=_run_query
    )

def get_all_tools(db: SQLDatabase, llm):
    """
    Return all custom SQL tools + RAG tool as a list.
    """
    tools = [
        list_tables_tool(db),
        get_schema_tool(db),
        sql_query_checker_tool(llm, db),
        run_query_tool(db),
        rag_tool,
    ]
    return tools
