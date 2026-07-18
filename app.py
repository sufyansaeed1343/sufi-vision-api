from flask import Flask, request, jsonify, send_file
import torch
from diffusers import StableDiffusionXLPipeline
import uuid

app = Flask(__name__)

# Load Stable Diffusion XL model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

pipe = pipe.to("cuda")

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
    filename = f"{uuid.uuid4()}.png"
    image.save(filename)

    # Return image directly
    return send_file(filename, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)