import os
from flask import Flask, request, jsonify, render_template
import requests
import concurrent.futures
import sqlite3

app = Flask(__name__)

# ============================
# 🔑 SUAS CHAVES DE API
# ============================


def get_api_key(env_var):
    """Recupera uma chave de API a partir das variáveis de ambiente."""
    value = os.getenv(env_var)
    if not value:
        raise RuntimeError(
            f"Variável de ambiente '{env_var}' não definida. "
            "Configure as chaves de API necessárias antes de executar o aplicativo."
        )
    return value


DEEPSEEK_API_KEY = get_api_key("DEEPSEEK_API_KEY")
GROK_API_KEY = get_api_key("GROK_API_KEY")
QWEN3_MAX_API_KEY = get_api_key("QWEN3_MAX_API_KEY")
OPENROUTER_API_KEY = get_api_key("OPENROUTER_API_KEY")

# ============================
# 🌐 ENDPOINTS
# ============================
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
GROK_API_URL = "https://api.grok.com/v1/chat/completions"
QWEN3_MAX_API_URL = "https://api.alibabacloud.com/modelstudio/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-4o-mini"

# ============================
# 📦 BANCO LOCAL
# ============================
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  pergunta TEXT,
                  deepseek TEXT,
                  grok TEXT,
                  qwen TEXT,
                  openrouter TEXT,
                  analise TEXT,
                  concordancia INTEGER,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()


def salvar_historico(pergunta, respostas, analise, concordancia):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''INSERT INTO history (pergunta, deepseek, grok, qwen, openrouter, analise, concordancia)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (pergunta, respostas.get('deepseek',''), respostas.get('grok',''),
               respostas.get('qwen3_max',''), respostas.get('openrouter',''),
               analise, concordancia))
    conn.commit()
    conn.close()


# ============================
# 🔧 FUNÇÃO GENÉRICA DE CHAMADA
# ============================
def call_api(url, api_key, model, prompt):
    """Faz chamada com cabeçalhos válidos para OpenRouter e debug."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Referer": "http://localhost",   # ⚡ o nome correto é "Referer", não "HTTP-Referer"
        "X-Title": "Moderador AI"
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    print(f"\n📡 Enviando para {url} | modelo: {model}")
    print(f"🧠 Prompt:\n{prompt[:300]}...\n")

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"🔁 Status: {response.status_code}")
        if response.status_code != 200:
            print("⚠️ Erro recebido:", response.text)
            return f"[Erro {response.status_code}] {response.text}"
        conteudo = response.json()["choices"][0]["message"]["content"]
        print(f"✅ Resposta: {conteudo[:100]}...\n")
        return conteudo
    except Exception as e:
        print(f"❌ Exceção:", e)
        return f"Erro: {e}"

# ============================
# 🧠 ANÁLISE AUTOMÁTICA
# ============================
def gerar_analise(pergunta, respostas):
    prompt = f"""
Você é um moderador especialista em análise comparativa entre inteligências artificiais.
Analise as seguintes respostas sobre a pergunta: "{pergunta}".

Resposta DeepSeek:
{respostas.get('deepseek', '')}

Resposta Grok:
{respostas.get('grok', '')}

Resposta Qwen3-Max:
{respostas.get('qwen3_max', '')}

Resposta OpenRouter:
{respostas.get('openrouter', '')}

Siga estas instruções:
1. Liste os pontos em comum.
2. Liste as divergências.
3. Avalie qual resposta é mais precisa e fundamentada.
4. Dê uma conclusão final com a resposta mais correta e explicação breve.
"""
    return call_api(OPENROUTER_API_URL, OPENROUTER_API_KEY, OPENROUTER_MODEL, prompt)


def calcular_concordancia(respostas):
    respostas_text = [r for r in respostas.values() if r and not r.startswith("Erro")]
    if len(respostas_text) < 2:
        return 0
    base = respostas_text[0].lower()
    iguais = sum(1 for r in respostas_text if base[:100] in r.lower() or r.lower()[:100] in base)
    return int((iguais / len(respostas_text)) * 100)


# ============================
# 🌍 ROTAS
# ============================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/admin")
def admin():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute("SELECT id, pergunta, analise, concordancia, timestamp FROM history ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return render_template("admin.html", rows=rows)


@app.route("/teste")
def teste():
    resposta = call_api(
        OPENROUTER_API_URL,
        OPENROUTER_API_KEY,
        OPENROUTER_MODEL,
        "Responda apenas com 'ok'."
    )
    return jsonify({"resultado": resposta})


@app.route("/analisar", methods=["POST"])
def analisar():
    data = request.json
    pergunta = data.get("pergunta", "")
    respostas = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            "deepseek": executor.submit(call_api, DEEPSEEK_API_URL, DEEPSEEK_API_KEY, "deepseek-chat", pergunta),
            "grok": executor.submit(call_api, GROK_API_URL, GROK_API_KEY, "grok-3o-mini", pergunta),
            "qwen3_max": executor.submit(call_api, QWEN3_MAX_API_URL, QWEN3_MAX_API_KEY, "alibaba/qwen3-max-instruct", pergunta),
            "openrouter": executor.submit(call_api, OPENROUTER_API_URL, OPENROUTER_API_KEY, OPENROUTER_MODEL, pergunta),
        }

        for nome, future in futures.items():
            try:
                respostas[nome] = future.result(timeout=35)
            except Exception as e:
                respostas[nome] = f"Erro em {nome}: {e}"

    analise = gerar_analise(pergunta, respostas)
    concordancia = calcular_concordancia(respostas)
    salvar_historico(pergunta, respostas, analise, concordancia)

    return jsonify({
        "respostas": respostas,
        "analise": analise,
        "concordancia": concordancia
    })


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
