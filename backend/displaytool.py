import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase

sql_db = SQLDatabase.from_uri("sqlite:///Chinook.db")
@tool("analytics_tool", return_direct=True)
def analytics_tool(query: str, chart_type: str) -> str:
    """
    Run a SQL query and return a Base64 encoded chart image.
    chart_type: 'bar', 'pie', or 'histogram'
    """
    try:
        # Get a raw SQLite connection
        conn = sql_db._engine.raw_connection()
        cursor = conn.cursor()

        # Execute SQL
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        conn.close()

        if not results:
            return "No data found."

        # Convert to DataFrame
        df = pd.DataFrame(results, columns=columns)

        plt.figure(figsize=(8, 5))

        # Histogram
        if chart_type.lower() == "histogram":
            numeric_cols = df.select_dtypes(include="number").columns
            if not len(numeric_cols):
                return "No numeric column found for histogram."
            col = numeric_cols[0]
            df[col].plot(kind="hist", bins=10, color='skyblue', edgecolor='black')
            plt.title(f"Histogram of {col}")

        # Pie chart
        elif chart_type.lower() == "pie":
            col = df.columns[0]
            df[col].value_counts().plot(kind="pie", autopct='%1.1f%%')
            plt.ylabel("")
            plt.title(f"Pie Chart of {col}")

        # Bar chart
        elif chart_type.lower() == "bar":
            if len(df.columns) < 2:
                return "Bar chart needs at least two columns."
            df.plot(kind="bar", x=df.columns[0], y=df.columns[1], color='skyblue')
            plt.title(f"{df.columns[1]} by {df.columns[0]}")

        else:
            return "Invalid chart_type."

        # Save to Base64
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        return f"Error generating chart: {str(e)}"
