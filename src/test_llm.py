from langchain_openai import ChatOpenAI
from src.config import GROQ_API_KEY,MODEL

llm =ChatOpenAI(
    model=MODEL,
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
) 

response=llm.invoke("Say hello in one short sentence.")
print(response.content)