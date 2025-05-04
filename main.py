# Install necessary packages
!pip install -q flask pyngrok diffusers transformers accelerate

# Import required modules
from flask import Flask, render_template_string, request
from pyngrok import ngrok

import torch
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64
from PIL import Image

# Load the Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.to("cuda")

# Define Flask app
app = Flask(__name__)

# HTML template inline (because Colab can't serve external HTML templates)
template = '''
<!doctype html>
<title>Stable Diffusion Generator</title>
<h1>Generate an Image</h1>
<form method="POST" action="/submit-prompt">
  <input name="prompt-input" type="text" placeholder="Enter prompt here" style="width:300px"/>
  <input type="submit"/>
</form>
{% if generated_image %}
  <h2>Generated Image:</h2>
  <img src="{{ generated_image }}">
{% endif %}
'''

@app.route('/')
def index():
    return render_template_string(template)

@app.route('/submit-prompt', methods=['POST'])
def generate():
    prompt = request.form['prompt-input']
    image = pipe(prompt).images[0]

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    img_str = f"data:image/png;base64,{img_str}"

    return render_template_string(template, generated_image=img_str)

# Start ngrok tunnel to the Flask app
public_url = ngrok.connect(5000)
print(f" * ngrok tunnel running at: {public_url}")

# Run the Flask app
app.run(port=5000)
