import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__, template_folder="templates")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-3.1-flash-lite"
CONFIG_PATH = Path(__file__).with_name("chatbot_config.txt")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured.")

client = genai.Client(api_key=API_KEY)
SYSTEM_PROMPT = CONFIG_PATH.read_text(encoding="utf-8")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )

        answer = (response.text or "").strip()

        if not answer:
            answer = "I couldn't generate a response. Please try again."

        return jsonify({"answer": answer})

    except Exception as exc:
        app.logger.exception("Gemini API error")
        return jsonify({
            "error": "Unable to get a response right now. Please check the API configuration and try again."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
