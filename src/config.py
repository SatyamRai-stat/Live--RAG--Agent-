import os 
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY") 
MODEL = "openai/gpt-oss-120b" 

HF_TOKEN=os.getenv("HF_TOKEN")
MODEL1="Qwen/Qwen2.5-0.5B-Instruct:featherless-ai"

DB_URI = os.getenv("DB_URI")