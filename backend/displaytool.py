# analytics_tool.py
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from langchain_core.tools import Tool
from langchain_community.utilities import SQLDatabase

# ======================================
# Shared SQL Database Connection
# ======================================
try:
    sql_db = SQLDatabase.from_uri("sqlite:///Chinook.db")
except Exception as e:
    raise RuntimeError(f"❌ Failed to connect to Chinook database: {e}")

# ======================================
# Analytics Tool Function
# ======================================
def analytics_tool_fn(query: str, chart_type: str) -> str:
    """
    Run a SQL query and return a Base64 encoded chart image.
    chart_type: 'bar', 'pie', or 'histogram'
    """
    print(f"\n📊 [DEBUG] Analytics tool called: chart_type={chart_type}, query={query}")

    try:
        # Use existing SQL connection
        conn = sql_db._engine.raw_connection()
        cursor = conn.cursor()

        # Execute SQL
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        conn.close()

        if not results:
            return "No data found."

        # Convert results to DataFrame
        df = pd.DataFrame(results, columns=columns)

        plt.figure(figsize=(8, 5))

        # ==========================
        # Chart Types
        # ==========================
        if chart_type.lower() == "histogram":
            numeric_cols = df.select_dtypes(include="number").columns
            if not len(numeric_cols):
                return "No numeric column found for histogram."
            col = numeric_cols[0]
            df[col].plot(kind="hist", bins=10, color="skyblue", edgecolor="black")
            plt.title(f"Histogram of {col}")

        elif chart_type.lower() == "pie":
            col = df.columns[0]
            df[col].value_counts().plot(kind="pie", autopct="%1.1f%%")
            plt.ylabel("")
            plt.title(f"Pie Chart of {col}")

        elif chart_type.lower() == "bar":
            if len(df.columns) < 2:
                return "Bar chart needs at least two columns."
            df.plot(kind="bar", x=df.columns[0], y=df.columns[1], color="skyblue")
            plt.title(f"{df.columns[1]} by {df.columns[0]}")

        else:
            return "Invalid chart_type."

        # ==========================
        # Save as Base64 PNG
        # ==========================
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        return f"Error generating chart: {str(e)}"

# ======================================
# LangChain Tool Definition
# ======================================
analytics_tool = Tool(
    name="analytics_tool",
    func=analytics_tool_fn,
    description=(
        "Generate a chart from a SQL query result. "
        "chart_type can be 'histogram', 'pie', or 'bar'. "
        "Returns a Base64-encoded PNG image."
    )
)
