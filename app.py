from flask import Flask, request, jsonify, send_file
import requests
import os
import io

app = Flask(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


@app.route("/")
def home():
    return "SUFI VISION AI API is running"


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}

    prompt = data.get("prompt", "A futuristic city")

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={"inputs": prompt},
        timeout=300
    )

    # Image returned
    if response.status_code == 200 and response.headers.get("content-type", "").startswith("image"):
        return send_file(
            io.BytesIO(response.content),
            mimetype=response.headers["content-type"]
        )

    # JSON error returned
    try:
        error = response.json()
    except Exception:
        error = {"error": response.text}

    return jsonify({
        "status": "failed",
        "huggingface_response": error
    }), response.status_code


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
