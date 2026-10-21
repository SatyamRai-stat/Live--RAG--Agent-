
from huggingface_hub import InferenceClient
from src.config import HF_TOKEN, MODEL1

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto",
)

response = client.chat_completion(
    model="meta-llama/Llama-3.2-3B:featherless-ai",
    messages=[
        {
            "role": "user",
            "content": "Say hello in one short sentence."
        }
    ],
    max_tokens=50,
)

print(response.choices[0].message.content)
