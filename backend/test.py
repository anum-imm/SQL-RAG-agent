from displaytool import analytics_tool
import base64

def is_base64_png(data: str) -> bool:
    try:
        decoded = base64.b64decode(data)
        return decoded.startswith(b'\x89PNG')  # PNG magic number
    except Exception:
        return False

# Test with a valid query
result = analytics_tool.invoke({
    "query": "SELECT Milliseconds FROM Track LIMIT 20",
    "chart_type": "histogram"
})

if is_base64_png(result):
    with open("test_chart.png", "wb") as f:
        f.write(base64.b64decode(result))
    print("✅ PNG saved as test_chart.png")
else:
    print("❌ Not a valid PNG:", result[:200])
