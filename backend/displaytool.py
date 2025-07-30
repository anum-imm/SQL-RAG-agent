import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase

# Connect to SQLite database
sql_db = SQLDatabase.from_uri("sqlite:///Chinook.db")

@tool("analytics_tool", return_direct=True)
def analytics_tool(query: str, chart_type: str) -> str:
    """
    Run a SQL query and return a Base64 encoded chart image.
    chart_type: 'bar', 'pie', or 'histogram'
    """
    try:
        # Run SQL query directly into a DataFrame
        df = pd.read_sql_query(query, sql_db._engine)

        # Handle empty result
        if df.empty:
            return "No data found."

        # Create figure
        plt.figure(figsize=(8, 5))

        # ========================
        # HISTOGRAM
        # ========================
        if chart_type.lower() == "histogram":
            numeric_cols = df.select_dtypes(include="number").columns
            if not len(numeric_cols):
                # Try to convert all columns to numeric
                df = df.apply(pd.to_numeric, errors="coerce")
                numeric_cols = df.select_dtypes(include="number").columns
                if not len(numeric_cols):
                    return "No numeric column found for histogram."

            col = numeric_cols[0]
            df[col].plot(kind="hist", bins=10, color='skyblue', edgecolor='black')
            plt.title(f"Histogram of {col}")

        # ========================
        # PIE CHART
        # ========================
        elif chart_type.lower() == "pie":
            col = df.columns[0]
            value_counts = df[col].value_counts()
            if value_counts.empty:
                return "No data available for pie chart."
            value_counts.plot(kind="pie", autopct='%1.1f%%')
            plt.ylabel("")
            plt.title(f"Pie Chart of {col}")

        # ========================
        # BAR CHART
        # ========================
        elif chart_type.lower() == "bar":
            if len(df.columns) < 2:
                return "Bar chart needs at least two columns."

            # Convert second column to numeric if possible
            df[df.columns[1]] = pd.to_numeric(df[df.columns[1]], errors="coerce")

            if df[df.columns[1]].isnull().all():
                return "No numeric data found for bar chart."

            df.plot(kind="bar", x=df.columns[0], y=df.columns[1], color='skyblue')
            plt.title(f"{df.columns[1]} by {df.columns[0]}")

        else:
            return "Invalid chart_type."

        # Save to Base64 string
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        return f"Error generating chart: {str(e)}"
