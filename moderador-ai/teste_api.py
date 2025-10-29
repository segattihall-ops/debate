import os
import requests

url = "https://openrouter.ai/api/v1/chat/completions"

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError(
        "Defina a variável de ambiente OPENROUTER_API_KEY antes de executar o teste."
    )

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "Referer": "http://localhost",  # cabeçalho correto exigido pelo OpenRouter
    "X-Title": "Moderador AI",
}

data = {
    "model": "openai/gpt-4o-mini",
    "messages": [
        {"role": "user", "content": "Olá, tudo bem? Responda com 'teste ok'."}
    ],
}

resp = requests.post(url, headers=headers, json=data, timeout=30)
print(resp.status_code)
print(resp.text)
