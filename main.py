from diffusers import DiffusionPipeline
import torch
import os
from PIL import Image
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__, template_folder='Web')

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']  # Get prompt from form input
    image = pipe(prompt).images[0]   # Generate image
    
    # Save the image to the static/images folder
    image_path = os.path.join('images', 'generated_image.png')
    image.save(image_path)
    
    # Return the generated image to display in the browser
    return render_template('index.html', prompt=prompt, image_file=image_path)

# Route for serving images
@app.route('/Images/<filename>')
def serve_image(filename):
    return send_from_directory('Images', filename)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)