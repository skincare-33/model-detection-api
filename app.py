from flask import Flask, request, send_file, jsonify
import os
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
import gc

app = Flask(__name__)

MODEL_PATH = r"best.pt"
IMG_SIZE = 640
CONF_THRESH = 0.01
MIN_CONF_TO_SAVE = 0.20

model = None

def load_model():
    global model
    if model is None:
        print(f"Loading model from: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        model.fuse()
        print(f"Model loaded. Classes: {model.names}\n")
    return model

def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    ww = (x2 - x1) / w
    hh = (y2 - y1) / h
    return xc, yc, ww, hh

def calculate_iou(box1, box2, orig_w, orig_h):
    x1_1 = (box1[1] - box1[3]/2) * orig_w
    y1_1 = (box1[2] - box1[4]/2) * orig_h
    x2_1 = (box1[1] + box1[3]/2) * orig_w
    y2_1 = (box1[2] + box1[4]/2) * orig_h
    
    x1_2 = (box2[1] - box2[3]/2) * orig_w
    y1_2 = (box2[2] - box2[4]/2) * orig_h
    x2_2 = (box2[1] + box2[3]/2) * orig_w
    y2_2 = (box2[2] + box2[4]/2) * orig_h
    
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

def process_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image data")
    
    m = load_model()
    
    try:
        results = m.predict(source=img, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False, device='cpu')
        r = results[0]
        orig_h, orig_w = r.orig_shape if hasattr(r, 'orig_shape') else (r.ori_shape[0], r.ori_shape[1])
        
        all_boxes = []
        save_boxes = []
        
        if hasattr(r, 'boxes') and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            
            for (bb, c, conf) in zip(xyxy, cls, confs):
                x1, y1, x2, y2 = bb
                xc, yc, ww, hh = xyxy_to_yolo(x1, y1, x2, y2, orig_w, orig_h)
                box_data = (int(c), xc, yc, ww, hh, float(conf))
                all_boxes.append(box_data)
                
                if conf >= MIN_CONF_TO_SAVE:
                    save_boxes.append(box_data)
        
        class_colors = {
            0: (0, 255, 0),
            1: (255, 0, 0),
            2: (0, 0, 255)
        }
        class_names = {0: 'low', 1: 'medium', 2: 'high'}
        
        filtered_boxes = []
        sorted_by_severity = sorted(all_boxes, key=lambda b: b[0], reverse=True)
        
        for box in sorted_by_severity:
            overlaps = False
            for filtered_box in filtered_boxes:
                iou = calculate_iou(box, filtered_box, orig_w, orig_h)
                if iou > 0.3:
                    if filtered_box[0] >= box[0]:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered_boxes.append(box)
        
        for b in filtered_boxes:
            c, xc, yc, w_rel, h_rel, conf = b
            x1 = int((xc - w_rel/2) * orig_w)
            y1 = int((yc - h_rel/2) * orig_h)
            x2 = int((xc + w_rel/2) * orig_w)
            y2 = int((yc + h_rel/2) * orig_h)
            
            color = class_colors.get(c, (128, 128, 128))
            thickness = 2 if conf >= MIN_CONF_TO_SAVE else 1
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            label = f"{class_names.get(c, 'unknown')}:{conf:.2f}"
            cv2.putText(img, label, (x1, max(10, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        _, buffer = cv2.imencode('.jpg', img)
        img_bytes = BytesIO(buffer)
        img_bytes.seek(0)
        
        metadata = {
            'total_boxes_detected': len(all_boxes),
            'total_boxes_saved': len(save_boxes),
            'filtered_boxes_shown': len(filtered_boxes)
        }
        
        if all_boxes:
            confs = [b[5] for b in all_boxes]
            metadata['confidence_stats'] = {
                'min': float(min(confs)),
                'max': float(max(confs)),
                'mean': float(np.mean(confs))
            }
        
        del results
        gc.collect()
        
        return img_bytes, metadata
        
    except Exception as e:
        gc.collect()
        raise e

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        image_bytes = file.read()
        
        result_image, metadata = process_image(image_bytes)
        
        return_metadata = request.form.get('return_metadata', 'false').lower() == 'true'
        
        if return_metadata:
            import base64
            result_image.seek(0)
            img_base64 = base64.b64encode(result_image.read()).decode('utf-8')
            
            return jsonify({
                'image': img_base64,
                'metadata': metadata
            })
        else:
            return send_file(
                result_image,
                mimetype='image/jpeg',
                as_attachment=False,
                download_name='detection_result.jpg'
            )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect_raw', methods=['POST'])
def detect_raw():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        image_bytes = file.read()
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Invalid image data")
        
        m = load_model()
        
        try:
            results = m.predict(source=img, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False, device='cpu')
            r = results[0]
            orig_h, orig_w = r.orig_shape if hasattr(r, 'orig_shape') else (r.ori_shape[0], r.ori_shape[1])
            
            all_boxes = []
            class_names = {0: 'low', 1: 'medium', 2: 'high'}
            
            if hasattr(r, 'boxes') and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                
                for (bb, c, conf) in zip(xyxy, cls, confs):
                    x1, y1, x2, y2 = bb
                    xc, yc, ww, hh = xyxy_to_yolo(x1, y1, x2, y2, orig_w, orig_h)
                    box_data = (int(c), xc, yc, ww, hh, float(conf))
                    all_boxes.append(box_data)
            
            filtered_boxes = []
            sorted_by_severity = sorted(all_boxes, key=lambda b: b[0], reverse=True)
            
            for box in sorted_by_severity:
                overlaps = False
                for filtered_box in filtered_boxes:
                    iou = calculate_iou(box, filtered_box, orig_w, orig_h)
                    if iou > 0.3:
                        if filtered_box[0] >= box[0]:
                            overlaps = True
                            break
                
                if not overlaps:
                    filtered_boxes.append(box)
            
            detections = []
            for b in filtered_boxes:
                c, xc, yc, w_rel, h_rel, conf = b
                x1 = round((xc - w_rel/2) * orig_w, 2)
                y1 = round((yc - h_rel/2) * orig_h, 2)
                x2 = round((xc + w_rel/2) * orig_w, 2)
                y2 = round((yc + h_rel/2) * orig_h, 2)
                
                detections.append({
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "class_id": c,
                    "class_name": class_names.get(c, 'unknown'),
                    "confidence": round(conf, 3)
                })
            
            del results
            gc.collect()
            
            return jsonify({
                "count": len(detections),
                "detections": detections
            })
            
        except Exception as e:
            gc.collect()
            raise e
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_classes': model.names if model else None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)