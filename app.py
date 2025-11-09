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
torch.set_num_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'

# Detection parameters (matching your working script)
IMG_SIZE = 1024
CONF_THRESH = 0.01  # Very low to detect everything
MIN_CONF_TO_RETURN = 0.20  # Only return boxes with conf >= this

# Load model once at startup
model = None

def load_model():
    global model
    if model is None:
        print("Loading YOLO model...")
        model = YOLO('best.pt')
        print("Model loaded successfully")
        print(f"Model classes: {model.names}")
    return model

def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    """Convert xyxy to YOLO format (normalized center coordinates)"""
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    ww = (x2 - x1) / w
    hh = (y2 - y1) / h
    return xc, yc, ww, hh

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
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

@app.route('/health', methods=['GET'])
def health_check():
    try:
        m = load_model()
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'model_classes': m.names,
            'detection_params': {
                'img_size': IMG_SIZE,
                'conf_threshold': CONF_THRESH,
                'min_conf_return': MIN_CONF_TO_RETURN
            }
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
            print(f"Received image from file: {image.size}, mode: {image.mode}")
        elif request.json and 'image' in request.json:
            # Base64 string
            img_data = base64.b64decode(request.json['image'])
            image = Image.open(BytesIO(img_data))
            print(f"Received image from base64: {image.size}, mode: {image.mode}")
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array for YOLO
        img_array = np.array(image)
        orig_h, orig_w = img_array.shape[:2]
        
        # Run inference with SAME parameters as your working script
        results = m.predict(
            source=img_array,
            imgsz=IMG_SIZE,      # 1024 like your script
            conf=CONF_THRESH,    # 0.01 like your script
            verbose=False
        )
        
        r = results[0]
        
        all_detections = []
        
        if hasattr(r, 'boxes') and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            
            # Collect all detections with their data
            for (bb, c, conf) in zip(xyxy, cls, confs):
                x1, y1, x2, y2 = bb
                
                # Convert to YOLO format
                xc, yc, ww, hh = xyxy_to_yolo(x1, y1, x2, y2, orig_w, orig_h)
                
                detection = {
                    'class_id': int(c),
                    'class_name': m.names[int(c)],
                    'confidence': round(float(conf), 3),
                    'bbox': {
                        'x1': round(float(x1), 2),
                        'y1': round(float(y1), 2),
                        'x2': round(float(x2), 2),
                        'y2': round(float(y2), 2)
                    },
                    'bbox_normalized': {
                        'xc': round(float(xc), 6),
                        'yc': round(float(yc), 6),
                        'w': round(float(ww), 6),
                        'h': round(float(hh), 6)
                    }
                }
                all_detections.append(detection)
            
            print(f"Raw detections: {len(all_detections)}")
        
        # Filter overlapping boxes: keep only highest severity
        # Sort by class descending (2=high, 1=medium, 0=low)
        sorted_detections = sorted(all_detections, key=lambda d: d['class_id'], reverse=True)
        
        filtered_detections = []
        for det in sorted_detections:
            # Check if this detection overlaps with any already filtered
            overlaps = False
            det_box = [det['bbox']['x1'], det['bbox']['y1'], 
                      det['bbox']['x2'], det['bbox']['y2']]
            
            for filtered_det in filtered_detections:
                filtered_box = [filtered_det['bbox']['x1'], filtered_det['bbox']['y1'],
                               filtered_det['bbox']['x2'], filtered_det['bbox']['y2']]
                iou = calculate_iou(det_box, filtered_box)
                
                # If IoU > 30%, consider them overlapping
                if iou > 0.3:
                    # If filtered box has higher or equal severity, skip current
                    if filtered_det['class_id'] >= det['class_id']:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered_detections.append(det)
        
        # Only return detections above minimum confidence
        final_detections = [
            d for d in filtered_detections 
            if d['confidence'] >= MIN_CONF_TO_RETURN
        ]
        
        print(f"Filtered detections: {len(filtered_detections)}")
        print(f"Final detections (conf >= {MIN_CONF_TO_RETURN}): {len(final_detections)}")
        
        # Calculate statistics
        if all_detections:
            all_confs = [d['confidence'] for d in all_detections]
            stats = {
                'raw_detections': len(all_detections),
                'filtered_detections': len(filtered_detections),
                'confidence_range': {
                    'min': round(min(all_confs), 3),
                    'max': round(max(all_confs), 3),
                    'mean': round(sum(all_confs) / len(all_confs), 3)
                }
            }
        else:
            stats = {
                'raw_detections': 0,
                'filtered_detections': 0,
                'confidence_range': None
            }
        
        response = {
            'success': True,
            'detections': final_detections,
            'count': len(final_detections),
            'image_size': {
                'width': orig_w,
                'height': orig_h
            },
            'stats': stats
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during detection: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e),
            'details': error_details
        }), 500

@app.route('/detect-raw', methods=['POST'])
def detect_acne_raw():
    """Return ALL detections without filtering (for debugging)"""
    try:
        m = load_model()
        
        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(file.stream)
        elif request.json and 'image' in request.json:
            img_data = base64.b64decode(request.json['image'])
            image = Image.open(BytesIO(img_data))
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        orig_h, orig_w = img_array.shape[:2]
        
        results = m.predict(
            source=img_array,
            imgsz=IMG_SIZE,
            conf=CONF_THRESH,
            verbose=False
        )
        
        r = results[0]
        all_detections = []
        
        if hasattr(r, 'boxes') and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            
            for (bb, c, conf) in zip(xyxy, cls, confs):
                x1, y1, x2, y2 = bb
                detection = {
                    'class_id': int(c),
                    'class_name': m.names[int(c)],
                    'confidence': round(float(conf), 3),
                    'bbox': {
                        'x1': round(float(x1), 2),
                        'y1': round(float(y1), 2),
                        'x2': round(float(x2), 2),
                        'y2': round(float(y2), 2)
                    }
                }
                all_detections.append(detection)
        
        return jsonify({
            'success': True,
            'detections': all_detections,
            'count': len(all_detections),
            'note': 'This endpoint returns ALL detections without filtering'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    load_model()  # Preload model
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)), debug=True)