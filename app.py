"""
Advanced Currency Detection Web Server
- Live webcam feed with green border detection
- 2-second detection before showing result
- Shows only highest confidence currency (Unknown if below 90%)
- Automatic capture and storage of detected images
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import os
import json
import threading
import base64
import time

app = Flask(__name__)
app.secret_key = 'currency_detection_secret_key'
CORS(app)

# Configuration
MODEL_PATH = 'models/currency_model.h5'
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.90  # Changed to 90% for showing valid currency
UNKNOWN_THRESHOLD = 0.60
CAPTURE_DIR = 'captured_images'
AUTO_CAPTURE_ENABLED = True
STABILIZATION_SECONDS = 2

# Global variables
model = None
class_names = ['Currency 1', 'Currency 2', 'Currency 3']
has_not_currency = False
camera = None
detection_stats = {
    'total_detections': 0,
    'avg_confidence': 0,
    'captures_saved': 0
}

# Detection tracking
detection_start_time = None
current_detection_class = None
latest_result = None
latest_result_image = None

# Create directories
os.makedirs(CAPTURE_DIR, exist_ok=True)
for class_name in class_names:
    os.makedirs(os.path.join(CAPTURE_DIR, class_name), exist_ok=True)

def load_model_and_classes():
    """Load the trained model and class names"""
    global model, class_names, has_not_currency
    
    print("Loading model...")
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✓ Model loaded successfully")
    else:
        print("❌ Model not found. Please train first.")
        return False
    
    if os.path.exists('models/class_names.txt'):
        with open('models/class_names.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"✓ Class names loaded: {class_names}")
        has_not_currency = 'not_currency' in class_names
    
    return True

def preprocess_frame(frame):
    """Preprocess frame for model prediction"""
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def detect_currency(frame):
    """Detect currency in frame and return predictions"""
    if model is None:
        return None
    
    processed = preprocess_frame(frame)
    predictions = model.predict(processed, verbose=0)[0]
    
    max_idx = np.argmax(predictions)
    max_conf = predictions[max_idx]
    predicted_class = class_names[max_idx]
    
    # Check if confidence is below 90% or it's not_currency
    is_unknown = False
    display_class = predicted_class
    
    if predicted_class == 'not_currency':
        is_unknown = True
        display_class = 'Unknown'
    elif max_conf < CONFIDENCE_THRESHOLD:  # Show Unknown if below 90%
        is_unknown = True
        display_class = 'Unknown'
    
    return {
        'class': display_class,
        'original_class': predicted_class if not is_unknown else 'unknown',
        'confidence': float(max_conf),
        'is_unknown': is_unknown,
        'all_predictions': {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    }

def draw_detection_overlay(frame, detection):
    """Draw bounding box overlay on frame"""
    height, width = frame.shape[:2]
    
    # Always show the guide box
    box_w = int(width * 0.7)
    box_h = int(height * 0.5)
    x1 = (width - box_w) // 2
    y1 = (height - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h
    
    if detection and detection['confidence'] >= CONFIDENCE_THRESHOLD and not detection.get('is_unknown', False):
        # Valid currency (>=90%) - GREEN box
        color = (0, 255, 100)
        thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Corner markers
        corner_len = 30
        corner_thick = 4
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, corner_thick)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, corner_thick)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, corner_thick)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, corner_thick)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, corner_thick)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, corner_thick)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, corner_thick)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, corner_thick)
        
        # Simple label
        label = f"{detection['class']}: {detection['confidence']*100:.1f}%"
        cv2.putText(frame, label, (x1 + 10, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        
    elif detection:
        # Unknown or below 90% - ORANGE box
        color = (0, 165, 255)
        thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        cv2.putText(frame, "Unknown", (x1 + 10, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    else:
        # No detection - gray dashed box
        dash_length = 15
        gap_length = 8
        color = (100, 100, 100)
        
        for x in range(x1, x2, dash_length + gap_length):
            cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, 2)
            cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, 2)
        
        for y in range(y1, y2, dash_length + gap_length):
            cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, 2)
            cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, 2)
    
    # Simple stats at top left
    cv2.putText(frame, f"Detected: {detection_stats['total_detections']}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def save_captured_image(frame, detection):
    """Save captured frame"""
    if not AUTO_CAPTURE_ENABLED or detection is None or detection.get('is_unknown', False):
        return False
    
    if detection['confidence'] >= CONFIDENCE_THRESHOLD:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        class_name = detection['original_class']
        confidence = int(detection['confidence'] * 100)
        
        filename = f"{timestamp}_conf{confidence}.jpg"
        filepath = os.path.join(CAPTURE_DIR, class_name, filename)
        
        cv2.imwrite(filepath, frame)
        detection_stats['captures_saved'] += 1
        return True
    
    return False

def generate_frames():
    """Generate video frames with detection overlay"""
    global camera, detection_stats, detection_start_time, current_detection_class, latest_result, latest_result_image
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Detect currency
        detection = detect_currency(frame)
        
        # Check for stable detection (only track valid currency with >=90% confidence)
        if detection and not detection.get('is_unknown', False) and detection['confidence'] >= CONFIDENCE_THRESHOLD:
            if current_detection_class == detection['original_class']:
                if detection_start_time is None:
                    detection_start_time = time.time()
                elif time.time() - detection_start_time >= STABILIZATION_SECONDS:
                    # 2 seconds passed - show result
                    if latest_result is None:  # Only show once
                        latest_result = detection
                        _, buffer = cv2.imencode('.jpg', frame)
                        latest_result_image = base64.b64encode(buffer).decode('utf-8')
                        
                        # Save for retraining
                        save_captured_image(frame, detection)
                        detection_stats['total_detections'] += 1
            else:
                # New detection
                current_detection_class = detection['original_class']
                detection_start_time = time.time()
        else:
            # No valid detection (below 90% or unknown)
            current_detection_class = None
            detection_start_time = None
        
        # Draw overlay
        frame = draw_detection_overlay(frame, detection)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html', class_names=class_names)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result')
def get_result():
    """Get the latest detection result"""
    global latest_result, latest_result_image
    
    if latest_result and latest_result_image:
        return jsonify({
            'has_result': True,
            'result': latest_result,
            'image': latest_result_image
        })
    
    return jsonify({'has_result': False})

@app.route('/clear_result', methods=['POST'])
def clear_result():
    """Clear the current result"""
    global latest_result, latest_result_image, detection_start_time, current_detection_class
    
    latest_result = None
    latest_result_image = None
    detection_start_time = None
    current_detection_class = None
    
    return jsonify({'status': 'cleared'})

@app.route('/stats')
def get_stats():
    """Get detection statistics"""
    return jsonify({
        'total_detections': detection_stats['total_detections'],
        'avg_confidence': detection_stats['avg_confidence'],
        'captures_saved': detection_stats['captures_saved']
    })

@app.route('/settings', methods=['POST'])
def update_settings():
    """Update detection settings"""
    global CONFIDENCE_THRESHOLD
    
    data = request.json
    if 'threshold' in data:
        CONFIDENCE_THRESHOLD = float(data['threshold'])
    
    return jsonify({'status': 'updated'})

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    """Handle uploaded image prediction"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        detection = detect_currency(img)
        
        if detection:
            predictions = []
            for class_name, conf in detection['all_predictions'].items():
                predictions.append({
                    'class': class_name,
                    'confidence': conf
                })
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'best_match': detection['class'],
                'confidence': detection['confidence'],
                'is_unknown': detection['is_unknown'],
                'image': img_base64
            })
        else:
            return jsonify({'error': 'Detection failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*50)
    print("CURRENCY DETECTION SYSTEM")
    print("="*50)
    
    if not load_model_and_classes():
        print("\n❌ Please train the model first")
        exit(1)
    
    print("\n🌐 Server running at: http://localhost:5000")
    print("📸 Hold currency for 2 seconds to detect")
    print("✅ Shows only if confidence >= 90%")
    print("❓ Shows Unknown if below 90%")
    print("Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)