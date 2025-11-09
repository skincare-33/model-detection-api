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

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter

# Load model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Configuration
IMG_SIZE = 1024
CONF_THRESH = 0.01
MIN_CONF_TO_SAVE = 0.20
IOU_THRESHOLD = 0.3

class_names = {0: 'low', 1: 'medium', 2: 'high'}

def calculate_iou(box1, box2, img_w, img_h):
    """Calculate IoU between two boxes in YOLO format (xc, yc, w, h)"""
    # box format: (class, xc, yc, w, h, conf)
    x1_1 = (box1[1] - box1[3]/2) * img_w
    y1_1 = (box1[2] - box1[4]/2) * img_h
    x2_1 = (box1[1] + box1[3]/2) * img_w
    y2_1 = (box1[2] + box1[4]/2) * img_h
    
    x1_2 = (box2[1] - box2[3]/2) * img_w
    y1_2 = (box2[2] - box2[4]/2) * img_h
    x2_2 = (box2[1] + box2[3]/2) * img_w
    y2_2 = (box2[2] + box2[4]/2) * img_h
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    """Convert xyxy format to YOLO format"""
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    ww = (x2 - x1) / w
    hh = (y2 - y1) / h
    return xc, yc, ww, hh

def filter_overlapping_boxes(boxes, img_w, img_h):
    """Filter overlapping boxes, keeping highest severity"""
    if not boxes:
        return []
    
    filtered_boxes = []
    # Sort by class descending (2=high, 1=medium, 0=low)
    sorted_by_severity = sorted(boxes, key=lambda b: b[0], reverse=True)
    
    for box in sorted_by_severity:
        overlaps = False
        for filtered_box in filtered_boxes:
            iou = calculate_iou(box, filtered_box, img_w, img_h)
            if iou > IOU_THRESHOLD:
                if filtered_box[0] >= box[0]:
                    overlaps = True
                    break
        
        if not overlaps:
            filtered_boxes.append(box)
    
    return filtered_boxes

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'message': 'Acne Detection API',
        'endpoints': {
            '/detect': 'POST - Detect acne in image',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/detect', methods=['POST'])
def detect_acne():
    try:
        # Check if image is in request
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Handle base64 image
        else:
            image_data = request.json['image']
            # Remove data:image/jpeg;base64, prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        orig_h, orig_w = img.shape[:2]
        
        # Run inference
        results = model.predict(
            source=img,
            imgsz=IMG_SIZE,
            conf=CONF_THRESH,
            verbose=False
        )
        
        r = results[0]
        all_boxes = []
        
        # Extract boxes
        if hasattr(r, 'boxes') and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            
            for (bb, c, conf) in zip(xyxy, cls, confs):
                x1, y1, x2, y2 = bb
                xc, yc, ww, hh = xyxy_to_yolo(x1, y1, x2, y2, orig_w, orig_h)
                all_boxes.append((int(c), xc, yc, ww, hh, float(conf)))
        
        # Filter overlapping boxes
        filtered_boxes = filter_overlapping_boxes(all_boxes, orig_w, orig_h)
        
        # Filter by confidence and format response
        response_boxes = []
        for box in filtered_boxes:
            c, xc, yc, w, h, conf = box
            if conf >= MIN_CONF_TO_SAVE:
                response_boxes.append({
                    'class': int(c),
                    'class_name': class_names.get(int(c), 'unknown'),
                    'confidence': round(float(conf), 3),
                    'bbox': {
                        'x_center': round(float(xc), 4),
                        'y_center': round(float(yc), 4),
                        'width': round(float(w), 4),
                        'height': round(float(h), 4)
                    },
                    # Also provide absolute coordinates
                    'bbox_absolute': {
                        'x': int((xc - w/2) * orig_w),
                        'y': int((yc - h/2) * orig_h),
                        'width': int(w * orig_w),
                        'height': int(h * orig_h)
                    }
                })
        
        # Calculate summary statistics
        summary = {
            'total_detections': len(response_boxes),
            'low': sum(1 for b in response_boxes if b['class'] == 0),
            'medium': sum(1 for b in response_boxes if b['class'] == 1),
            'high': sum(1 for b in response_boxes if b['class'] == 2)
        }
        
        return jsonify({
            'success': True,
            'image_size': {'width': orig_w, 'height': orig_h},
            'detections': response_boxes,
            'summary': summary
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)