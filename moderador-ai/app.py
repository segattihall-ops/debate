import os
import json
from flask import Flask, request, jsonify, render_template
import requests
import concurrent.futures
import sqlite3

app = Flask(__name__)

# ============================
# Configurações de chaves de API
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
# Configuração de endpoints
# ============================
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
GROK_API_URL = "https://api.grok.com/v1/chat/completions"
QWEN3_MAX_API_URL = "https://api.alibabacloud.com/modelstudio/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-4o-mini"

# ============================
# Banco local
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
# Função genérica de chamada
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

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code != 200:
            return f"[Erro {response.status_code}] {response.text}"
        conteudo = response.json()["choices"][0]["message"]["content"]
        return conteudo
    except Exception as e:
        return f"Erro: {e}"

# ============================
# Rotinas de análise
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


def gerar_interpretacao(texto, idioma, entonacao):
    idiomas = {
        "pt": {
            "nome": "portuguese",
            "detalhe": "Use português claro e natural com terminologia técnica quando necessário."
        },
        "en": {
            "nome": "english",
            "detalhe": "Write in concise international English suitable for technical documentation."
        },
        "es": {
            "nome": "spanish",
            "detalhe": "Use espanhol neutro, preciso e apropriado para times de tecnologia."
        },
    }

    info_idioma = idiomas.get(idioma, idiomas["pt"])

    prompt = f"""
Act as a command interpreter that converts natural language feature requests into precise technical instructions before execution.

Input request: "{texto.strip()}"
Desired output language: {info_idioma['nome']}.
Language guidance: {info_idioma['detalhe']}
Tone guidance: {entonacao}.

Return only a valid JSON object with the following structure:
{{
  "comando": "...",
  "variacoes": ["...", "...", "..."]
}}

Requirements:
- The field "comando" must contain a single, complete instruction ready to be executed or handed to a development team.
- Capture the intent, context, constraints, and relevant interface or implementation details explicitly.
- Provide three alternative phrasings in the "variacoes" array, in the same language, each maintaining the intent while exploring different wording.
- Respect the requested tone while keeping the message objective and technically actionable.
- Do not include backticks or any additional commentary outside the JSON.
"""

    resposta = call_api(OPENROUTER_API_URL, OPENROUTER_API_KEY, OPENROUTER_MODEL, prompt)

    try:
        interpretacao = json.loads(resposta)
    except json.JSONDecodeError:
        interpretacao = {
            "comando": resposta.strip(),
            "variacoes": []
        }

    comando = interpretacao.get("comando", "").strip()
    variacoes = interpretacao.get("variacoes", [])
    if isinstance(variacoes, str):
        variacoes = [variacoes.strip()] if variacoes.strip() else []
    elif isinstance(variacoes, list):
        variacoes = [str(item).strip() for item in variacoes if str(item).strip()]
    else:
        variacoes = []

    return {
        "comando": comando,
        "variacoes": variacoes
    }


# ============================
# Rotas
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


@app.route("/interpretar", methods=["POST"])
def interpretar():
    data = request.json or {}
    texto = (data.get("texto") or "").strip()
    idioma = (data.get("idioma") or "pt").lower()
    entonacao = (data.get("entonacao") or "neutro").strip()

    if not texto:
        return jsonify({"erro": "Forneça um texto para interpretação."}), 400

    resultado = gerar_interpretacao(texto, idioma, entonacao)
    return jsonify(resultado)


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
