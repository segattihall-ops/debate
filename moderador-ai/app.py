import os
import json
import re
import time
from pathlib import Path
from flask import Flask, request, jsonify, render_template, abort
import requests
import concurrent.futures
import sqlite3
import numpy as np
from numpy.linalg import norm

app = Flask(__name__)


def load_env_file():
    """Carrega variáveis de ambiente de um arquivo .env local, se disponível."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()


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
                  votos_sim INTEGER DEFAULT 0,
                  votos_nao INTEGER DEFAULT 0,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    c.execute("PRAGMA table_info(history)")
    colunas = {row[1] for row in c.fetchall()}
    if 'votos_sim' not in colunas:
        c.execute("ALTER TABLE history ADD COLUMN votos_sim INTEGER DEFAULT 0")
    if 'votos_nao' not in colunas:
        c.execute("ALTER TABLE history ADD COLUMN votos_nao INTEGER DEFAULT 0")

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
    last_id = c.lastrowid
    conn.commit()
    conn.close()
    return last_id


# ============================
# Função genérica de chamada
# ============================
def call_api(url, api_key, model, prompt, max_retries=3, initial_delay=2):
    """Faz chamada à API com lógica de re-tentativa e backoff exponencial."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Referer": "http://localhost",
        "X-Title": "Moderador AI"
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                conteudo = response.json()["choices"][0]["message"]["content"]
                return conteudo
            if 500 <= response.status_code < 600:
                raise requests.exceptions.HTTPError(f"Erro de servidor: {response.status_code}")
            return f"[Erro {response.status_code}] {response.text}"
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as err:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            return f"Erro: Falha persistente após {max_retries} tentativas. Último erro: {err}"
        except Exception as err:
            return f"Erro Crítico: {err}"

    return "Erro: Falha na comunicação após todas as tentativas."


def extrair_json(texto):
    if not texto:
        return "{}"
    match = re.search(r"\{.*\}", texto, re.DOTALL)
    return match.group(0) if match else "{}"


def get_semantic_vector_proxy(texto):
    prompt = f"""
    Analise exclusivamente o texto a seguir. Retorne apenas um objeto JSON
    com a chave "vector" contendo um array com cinco valores float entre 0.0 e 1.0.
    Cada posição representa respectivamente: Fato, Opinião, Linguagem Técnica, Detalhe, Conclusão.
    Não inclua explicações adicionais.

    TEXTO: <resposta>{texto}</resposta>
    """

    resposta_json_str = call_api(
        OPENROUTER_API_URL,
        OPENROUTER_API_KEY,
        OPENROUTER_MODEL,
        prompt,
    )

    try:
        bloco_json = extrair_json(resposta_json_str)
        dados = json.loads(bloco_json)
        vetor = dados.get("vector")
        if isinstance(vetor, list) and len(vetor) == 5:
            return np.array(vetor, dtype=float)
    except Exception as err:
        print(f"Erro ao gerar vetor semântico: {err}")
    return None


def cosine_similarity(A, B):
    if A is None or B is None:
        return 0.0
    denominador = norm(A) * norm(B)
    if denominador == 0:
        return 0.0
    return float(np.dot(A, B) / denominador)

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
    respostas_validas = [r for r in respostas.values() if r and not r.startswith(("Erro", "[Erro"))]
    if len(respostas_validas) < 2:
        return 0

    vetores = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(respostas_validas)) as executor:
        futuros = [executor.submit(get_semantic_vector_proxy, texto) for texto in respostas_validas]
        for futuro in concurrent.futures.as_completed(futuros):
            vetor = futuro.result()
            if vetor is not None:
                vetores.append(vetor)

    if len(vetores) < 2:
        return 0

    base = vetores[0]
    similaridades = [cosine_similarity(base, vetor) for vetor in vetores[1:]]
    if not similaridades:
        return 0
    media = np.mean(similaridades)
    return int(max(0, min(media, 1)) * 100)


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
    c.execute("SELECT id, pergunta, analise, concordancia, timestamp, votos_sim, votos_nao FROM history ORDER BY id DESC LIMIT 50")
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
                respostas[nome] = future.result(timeout=45)
            except Exception as e:
                respostas[nome] = f"Erro em {nome}: {e}"

    analise = gerar_analise(pergunta, respostas)
    concordancia = calcular_concordancia(respostas)
    registro_id = salvar_historico(pergunta, respostas, analise, concordancia)

    return jsonify({
        "respostas": respostas,
        "analise": analise,
        "concordancia": concordancia,
        "registro_id": registro_id
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


@app.route("/votar", methods=["POST"])
def votar():
    data = request.json or {}
    registro_id = data.get("id")
    voto_tipo = data.get("voto")

    if not registro_id or voto_tipo not in {"sim", "nao"}:
        return jsonify({"success": False, "message": "ID ou voto inválido."}), 400

    coluna = "votos_sim" if voto_tipo == "sim" else "votos_nao"

    try:
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        c.execute(f"UPDATE history SET {coluna} = {coluna} + 1 WHERE id = ?", (registro_id,))
        if c.rowcount == 0:
            conn.close()
            return jsonify({"success": False, "message": "Registro não encontrado."}), 404
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as err:
        return jsonify({"success": False, "message": str(err)}), 500


@app.route("/history/<int:registro_id>")
def obter_historico(registro_id):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute(
        """SELECT id, pergunta, deepseek, grok, qwen, openrouter, analise, concordancia, votos_sim, votos_nao, timestamp
            FROM history WHERE id = ?""",
        (registro_id,),
    )
    row = c.fetchone()
    conn.close()

    if not row:
        abort(404)

    return jsonify({
        "id": row[0],
        "pergunta": row[1],
        "deepseek": row[2],
        "grok": row[3],
        "qwen": row[4],
        "openrouter": row[5],
        "analise": row[6],
        "concordancia": row[7],
        "votos_sim": row[8],
        "votos_nao": row[9],
        "timestamp": row[10],
    })


init_db()


if __name__ == "__main__":
    app.run(debug=True)
