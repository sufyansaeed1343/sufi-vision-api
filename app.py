from flask import Flask, request, jsonify, send_file
import torch
from diffusers import StableDiffusionXLPipeline
import uuid
import os

app = Flask(__name__)

# Load Stable Diffusion XL model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# Use CPU on Railway (GPU available ho to automatically use ho jayega)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype
)

pipe = pipe.to(device)

@app.route('/')
def home():
    return "SUFI VISION AI API is running"

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', 'A futuristic city')

    # Generate image
    image = pipe(prompt).images[0]

    # Unique filename
    filename = f"/tmp/{uuid.uuid4()}.png"
    image.save(filename)

    return send_file(filename, mimetype='image/png')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
