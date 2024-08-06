import numpy as np
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
from huggingface_hub import login

# Configure the Hugging Face token
hf_token = "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"
login(token=hf_token)

# Load the depth estimation pipeline using the correct model
depth_estimator = pipeline(task="depth-estimation", model="Intel/zoedepth-nyu-kitti")

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        image = Image.open(file)
        ref_points = request.json['ref_points']
        known_distance_meters = request.json['known_distance_meters']

        # Perform depth estimation
        outputs = depth_estimator(image)
        depth = outputs['depth']

        # Convert depth to a PIL image
        depth_image = Image.fromarray((depth * 255).astype(np.uint8))

        # Get the coordinates of the reference points from the user
        ref_point1, ref_point2 = ref_points

        # Depth values at the reference points
        depth1 = depth_image.getpixel(ref_point1)
        depth2 = depth_image.getpixel(ref_point2)

        # Calculate the depth difference in pixel values
        depth_diff = abs(depth1 - depth2)

        # Calculate the scaling factor (meters per depth unit)
        scaling_factor = known_distance_meters / depth_diff

        # Convert the depth map to real-world distances
        depth_array = np.array(depth_image)
        real_world_depth = depth_array * scaling_factor

        # Save the real-world depth map as an image file
        real_world_depth_image = Image.fromarray((real_world_depth * 255).astype(np.uint8))
        output = io.BytesIO()
        real_world_depth_image.save(output, format='PNG')
        output.seek(0)

        return send_file(output, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
