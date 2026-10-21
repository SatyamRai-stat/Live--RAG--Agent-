from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from src.config import GROQ_API_KEY,MODEL

@tool
def get_unread_count(label:str)->int:
    """Returns the number of unread emails in a Gmail label."""
    return 5  # fake answer, just for testing

llm = ChatOpenAI(
    model=MODEL,
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

llm_with_tools=llm.bind_tools([get_unread_count])

response = llm_with_tools.invoke("How many unread emails do I have in my INBOX?")

print("Text reply:", response.content)
print("Tool calls:", response.tool_calls)