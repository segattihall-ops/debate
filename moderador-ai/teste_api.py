"""Script simples para testar a integração com o OpenRouter."""
import os
import requests


def main():
    url = "https://openrouter.ai/api/v1/chat/completions"

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Defina a variável de ambiente OPENROUTER_API_KEY antes de executar o teste."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Referer": "http://localhost",
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


if __name__ == "__main__":
    main()
