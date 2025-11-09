# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
from io import BytesIO
from PIL import Image
import torch

app = Flask(__name__)
CORS(app)

# Memory optimization
torch.set_num_threads(1)  # Reduce CPU threads
os.environ['OMP_NUM_THREADS'] = '1'

# Load model once at startup
model = None

def load_model():
    global model
    if model is None:
        print("Loading YOLO model...")
        model = YOLO('best.pt')
        model.fuse()  # Fuse model for faster inference
        print("Model loaded successfully")
    return model

@app.route('/health', methods=['GET'])
def health_check():
    try:
        m = load_model()
        return jsonify({
            'status': 'healthy',
            'model_loaded': True
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'error': str(e)
        }), 500

@app.route('/detect', methods=['POST'])
def detect_acne():
    try:
        # Load model
        m = load_model()
        
        # Get image from request
        if 'image' in request.files:
            # File upload
            file = request.files['image']
            image = Image.open(file.stream)
        elif 'image' in request.json:
            # Base64 string
            img_data = base64.b64decode(request.json['image'])
            image = Image.open(BytesIO(img_data))
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Resize image if too large (save memory)
        max_size = 640
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Run inference with lower image size
        results = m(img_array, imgsz=320, verbose=False)  # Reduced from 640 to 320
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detections.append({
                    'class_id': cls,
                    'class_name': m.names[cls],
                    'confidence': conf,
                    'bbox': {
                        'x1': x1, 'y1': y1,
                        'x2': x2, 'y2': y2
                    }
                })
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections)
        }), 200
        
    except Exception as e:
        print(f"Error during detection: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    load_model()  # Preload model
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))