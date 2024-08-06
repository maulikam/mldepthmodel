import os
import numpy as np
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import io
from huggingface_hub import login
import logging

# Configure the Hugging Face token
hf_token = "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"
login(token=hf_token)

# Specify the cache directory
cache_dir = "./model_cache"

# Ensure the cache directory exists
os.makedirs(cache_dir, exist_ok=True)

# Load the depth estimation pipeline using the correct model
depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large", cache_dir=cache_dir)

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    logging.debug("Upload endpoint called")
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    try:
        image = Image.open(file)
        logging.debug(f"Image opened successfully: {image}")
    except Exception as e:
        logging.error(f"Error opening image: {e}")
        return jsonify({'error': 'Invalid image file'}), 400

    try:
        ref_points = eval(request.form['ref_points'])
        known_distance_meters = float(request.form['known_distance_meters'])
        logging.debug(f"Reference points: {ref_points}, Known distance: {known_distance_meters}")
    except Exception as e:
        logging.error(f"Error parsing form data: {e}")
        return jsonify({'error': 'Invalid form data'}), 400

    try:
        # Convert reference points to integer coordinates
        ref_point1 = (int(ref_points[0]['x']), int(ref_points[0]['y']))
        ref_point2 = (int(ref_points[1]['x']), int(ref_points[1]['y']))
        logging.debug(f"Integer reference points: {ref_point1}, {ref_point2}")

        # Perform depth estimation
        outputs = depth_estimator(image)
        logging.debug(f"Depth estimation outputs: {outputs}")
        depth = outputs['depth']

        # Convert depth to a NumPy array
        depth_array = np.array(depth)
        logging.debug(f"Depth array shape: {depth_array.shape}")

        # Depth values at the reference points
        depth1 = depth_array[ref_point1[1], ref_point1[0]]
        depth2 = depth_array[ref_point2[1], ref_point2[0]]
        logging.debug(f"Depth values at reference points: {depth1}, {depth2}")

        # Calculate the depth difference in pixel values
        depth_diff = abs(depth1 - depth2)
        logging.debug(f"Depth difference: {depth_diff}")

        # Calculate the scaling factor (meters per depth unit)
        scaling_factor = known_distance_meters / depth_diff
        logging.debug(f"Scaling factor: {scaling_factor}")

        # Convert the depth map to real-world distances
        real_world_depth = depth_array * scaling_factor
        logging.debug(f"Real-world depth array: {real_world_depth}")

        # Save the real-world depth map as an image file
        real_world_depth_image = Image.fromarray((real_world_depth * 255).astype(np.uint8))
        output = io.BytesIO()
        real_world_depth_image.save(output, format='PNG')
        output.seek(0)

        return send_file(output, mimetype='image/png')
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({'error': 'Error processing image'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)