import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase

# Connect to SQLite database
sql_db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Set Seaborn theme for clean visuals
sns.set_theme(style="whitegrid")

@tool("analytics_tool", return_direct=True)
def analytics_tool(query: str, chart_type: str) -> str:
    """
    Run a valid SQL query (in standard SQL syntax) on the Chinook database and return a Base64-encoded chart image.
    chart_type: 'bar', 'pie', or 'histogram'
    """
    try:
        # Run SQL query into DataFrame
        df = pd.read_sql_query(query, sql_db._engine)

        if df.empty:
            return "No data found."

        # Create figure
        plt.figure(figsize=(10, 6), dpi=120)

        # ========================
        # HISTOGRAM
        # ========================
        if chart_type.lower() == "histogram":
            numeric_cols = df.select_dtypes(include="number").columns
            if not len(numeric_cols):
                df = df.apply(pd.to_numeric, errors="coerce")
                numeric_cols = df.select_dtypes(include="number").columns
                if not len(numeric_cols):
                    return "No numeric column found for histogram."

            col = numeric_cols[0]
            sns.histplot(df[col], bins=12, kde=True, color=sns.color_palette("husl", 1)[0])
            plt.title(f"Histogram of {col}", fontsize=14, fontweight="bold")

        # ========================
        # PIE CHART
        # ========================
        elif chart_type.lower() == "pie":
            col = df.columns[0]
            value_counts = df[col].value_counts()
            if value_counts.empty:
                return "No data available for pie chart."

            colors = sns.color_palette("husl", len(value_counts))
            wedges, texts, autotexts = plt.pie(
                value_counts,
                labels=value_counts.index,
                autopct='%1.1f%%',
                colors=colors,
                wedgeprops={"edgecolor": "white"}
            )
            plt.setp(autotexts, size=10, weight="bold", color="black")
            plt.title(f"Pie Chart of {col}", fontsize=14, fontweight="bold")

        # ========================
        # BAR CHART
        # ========================
        elif chart_type.lower() == "bar":
            if len(df.columns) < 2:
                return "Bar chart needs at least two columns."

            # Ensure numeric y values
            df[df.columns[1]] = pd.to_numeric(df[df.columns[1]].astype(str).str.strip(), errors="coerce")

            if df[df.columns[1]].isnull().all():
                return "No numeric data found for bar chart."

            # Sort bars by value
            df = df.sort_values(by=df.columns[1], ascending=False)

            colors = sns.color_palette("husl", len(df))
            ax = sns.barplot(
                x=df.columns[0],
                y=df.columns[1],
                data=df,
                palette=colors
            )

            # Add value labels on bars
            for p in ax.patches:
                ax.annotate(
                    f'{p.get_height():.0f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='black'
                )

            plt.xticks(rotation=30, ha="right")
            plt.title(f"{df.columns[1]} by {df.columns[0]}", fontsize=14, fontweight="bold")

        else:
            return "Invalid chart_type."

        # Save chart as Base64 string
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        return f"Error generating chart: {str(e)}"
