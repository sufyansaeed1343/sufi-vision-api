from flask import Flask, request, send_file, jsonify
import requests
import os
import io

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


@app.route("/")
def home():
    return "SUFI VISION AI API is running"


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()

    prompt = data.get("prompt", "A futuristic city")

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": prompt},
        timeout=300
    )

    if response.status_code != 200:
        return jsonify({
            "error": response.text
        }), 500

    return send_file(
        io.BytesIO(response.content),
        mimetype="image/png"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
