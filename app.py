"""
Advanced Currency Detection Web Server
- Capture mode: Detects for 2 seconds before showing result
- Shows result, then clears for next detection
- Web interface accessible in browser
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
import time
import base64

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = 'models/currency_model.h5'
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.85
UNKNOWN_THRESHOLD = 0.85
MIN_CONFIDENCE_GAP = 0.30
DETECTION_STABLE_TIME = 2.0  # Must detect same thing for 2 seconds

# Global variables
model = None
class_names = ['Currency 1', 'Currency 2', 'Currency 3']
has_not_currency = False
camera = None
detection_stats = {
    'total_detections': 0,
    'avg_confidence': 0
}

# Detection state management
detection_state = {
    'is_detecting': False,
    'current_class': None,
    'detection_start_time': None,
    'stable_count': 0,
    'last_result': None,
    'result_image': None,
    'show_result': False,
    'result_timestamp': None
}

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
        if has_not_currency:
            print("✓ Model has 'not_currency' class - will detect invalid images")
    
    return True

def preprocess_frame(frame):
    """Preprocess frame for model prediction"""
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def detect_currency(frame):
    """Detect currency in frame and return predictions with strict validation"""
    if model is None:
        return None
    
    processed = preprocess_frame(frame)
    predictions = model.predict(processed, verbose=0)[0]
    
    pred_indices = np.argsort(predictions)[::-1]
    top1_idx = pred_indices[0]
    top2_idx = pred_indices[1] if len(pred_indices) > 1 else top1_idx
    
    top1_conf = predictions[top1_idx]
    top2_conf = predictions[top2_idx]
    
    predicted_class = class_names[top1_idx]
    confidence_gap = top1_conf - top2_conf
    
    is_unknown = False
    display_class = predicted_class
    reason = ""
    
    # Detection validation rules
    if predicted_class == 'not_currency':
        is_unknown = True
        display_class = 'Unknown'
        reason = 'Not Currency'
    elif top1_conf < UNKNOWN_THRESHOLD:
        is_unknown = True
        display_class = 'Unknown'
        reason = f'Low Confidence ({top1_conf*100:.1f}%)'
    elif confidence_gap < MIN_CONFIDENCE_GAP:
        is_unknown = True
        display_class = 'Unknown'
        reason = f'Uncertain (Gap: {confidence_gap*100:.1f}%)'
    elif has_not_currency and 'not_currency' in class_names:
        not_curr_idx = class_names.index('not_currency')
        not_curr_conf = predictions[not_curr_idx]
        if not_curr_idx in [top1_idx, top2_idx]:
            is_unknown = True
            display_class = 'Unknown'
            reason = f'Possibly Not Currency'
        elif not_curr_conf > 0.20:
            is_unknown = True
            display_class = 'Unknown'
            reason = f'Mixed Signal'
    
    std_dev = np.std(predictions)
    if std_dev < 0.15:
        is_unknown = True
        display_class = 'Unknown'
        reason = 'No Clear Match'
    
    return {
        'class': display_class,
        'class_idx': int(top1_idx),
        'confidence': float(top1_conf),
        'confidence_gap': float(confidence_gap),
        'is_unknown': is_unknown,
        'unknown_reason': reason,
        'original_class': predicted_class,
        'all_predictions': {class_names[i]: float(predictions[i]) for i in range(len(class_names))},
        'top2_gap': float(confidence_gap)
    }

def process_detection(frame, detection):
    """Process detection and manage 2-second stable detection logic"""
    global detection_state
    
    current_time = time.time()
    
    # If showing result, check if we should clear it
    if detection_state['show_result']:
        # Show result for 5 seconds, then clear
        if current_time - detection_state['result_timestamp'] > 5.0:
            detection_state['show_result'] = False
            detection_state['last_result'] = None
            detection_state['result_image'] = None
            detection_state['is_detecting'] = False
            detection_state['current_class'] = None
            detection_state['detection_start_time'] = None
        return
    
    # Check if we have a valid detection
    if detection and detection['confidence'] >= CONFIDENCE_THRESHOLD and not detection.get('is_unknown', False):
        detected_class = detection['class']
        
        # First detection or new class detected
        if detection_state['current_class'] != detected_class:
            detection_state['current_class'] = detected_class
            detection_state['detection_start_time'] = current_time
            detection_state['is_detecting'] = True
        else:
            # Same class detected, check if stable for 2 seconds
            time_elapsed = current_time - detection_state['detection_start_time']
            
            if time_elapsed >= DETECTION_STABLE_TIME:
                # Stable detection for 2 seconds! Capture result
                detection_state['last_result'] = detection
                
                # Save the frame
                _, buffer = cv2.imencode('.jpg', frame)
                detection_state['result_image'] = base64.b64encode(buffer).decode('utf-8')
                
                detection_state['show_result'] = True
                detection_state['result_timestamp'] = current_time
                
                # Update stats
                detection_stats['total_detections'] += 1
                prev_avg = detection_stats['avg_confidence']
                n = detection_stats['total_detections']
                detection_stats['avg_confidence'] = (prev_avg * (n-1) + detection['confidence']) / n
                
                # Reset detection state
                detection_state['is_detecting'] = False
                detection_state['current_class'] = None
    else:
        # No valid detection or unknown
        detection_state['is_detecting'] = False
        detection_state['current_class'] = None
        detection_state['detection_start_time'] = None

def draw_detection_guide(frame):
    """Draw guide frame for positioning currency"""
    height, width = frame.shape[:2]
    
    box_w = int(width * 0.7)
    box_h = int(height * 0.5)
    x1 = (width - box_w) // 2
    y1 = (height - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h
    
    # Dashed rectangle guide
    dash_length = 20
    gap_length = 10
    color = (100, 100, 100)
    
    for x in range(x1, x2, dash_length + gap_length):
        cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, 2)
        cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, 2)
    
    for y in range(y1, y2, dash_length + gap_length):
        cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, 2)
        cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, 2)
    
    # Instructions
    if detection_state['is_detecting']:
        # Detecting - show progress
        time_elapsed = time.time() - detection_state['detection_start_time']
        progress = min(time_elapsed / DETECTION_STABLE_TIME, 1.0)
        
        # Draw progress bar
        bar_width = 300
        bar_height = 30
        bar_x = (width - bar_width) // 2
        bar_y = y2 + 60
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        # Progress
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 100), -1)
        
        # Text
        text = f"Detecting: {detection_state['current_class']}..."
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = bar_y - 10
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        
        # Percentage
        pct_text = f"{int(progress * 100)}%"
        pct_size = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        pct_x = bar_x + (bar_width - pct_size[0]) // 2
        pct_y = bar_y + 22
        cv2.putText(frame, pct_text, (pct_x, pct_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        # Waiting for detection
        text = "Hold currency steady in frame"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = y2 + 50
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    return frame

def generate_frames():
    """Generate video frames with detection overlay"""
    global camera, detection_state
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Always show camera feed (result shown in separate div)
        # Detect currency
        detection = detect_currency(frame)
        
        # Process detection (2-second stability check)
        if not detection_state['show_result']:
            process_detection(frame, detection)
        
        # Draw guide frame
        frame = draw_detection_guide(frame)
        
        # Add stats overlay
        stats_text = [
            f"Total Detections: {detection_stats['total_detections']}",
            f"Avg Confidence: {detection_stats['avg_confidence']*100:.1f}%"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html', class_names=class_names)

@app.route('/guide')
def guide():
    """Serve the how-to-use guide page"""
    return render_template('guide.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """Get detection statistics"""
    return jsonify(detection_stats)

@app.route('/result')
def get_result():
    """Get current detection result"""
    if detection_state['show_result'] and detection_state['last_result']:
        return jsonify({
            'has_result': True,
            'result': detection_state['last_result'],
            'image': detection_state['result_image']
        })
    else:
        return jsonify({'has_result': False})

@app.route('/clear_result', methods=['POST'])
def clear_result():
    """Manually clear result"""
    detection_state['show_result'] = False
    detection_state['last_result'] = None
    detection_state['result_image'] = None
    detection_state['is_detecting'] = False
    detection_state['current_class'] = None
    return jsonify({'status': 'cleared'})

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
        
        detection = detect_currency(img)
        
        if detection:
            predictions = []
            for class_name, conf in detection['all_predictions'].items():
                predictions.append({
                    'class': class_name,
                    'confidence': conf
                })
            
            if detection['confidence'] >= CONFIDENCE_THRESHOLD and not detection.get('is_unknown', False):
                detection_stats['total_detections'] += 1
                prev_avg = detection_stats['avg_confidence']
                n = detection_stats['total_detections']
                detection_stats['avg_confidence'] = (prev_avg * (n-1) + detection['confidence']) / n
            
            # Encode image
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'best_match': detection['class'],
                'confidence': detection['confidence'],
                'is_unknown': detection.get('is_unknown', False),
                'image': img_base64
            })
        else:
            return jsonify({'error': 'Detection failed'}), 500
            
    except Exception as e:
        print(f"Upload prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("CURRENCY DETECTION - CAPTURE MODE")
    print("=" * 60)
    
    if not load_model_and_classes():
        print("\n❌ Please train the model first: python train.py")
        exit(1)
    
    print("\n" + "=" * 60)
    print("Starting web server...")
    print("=" * 60)
    print("\n🌐 Open your browser and go to:")
    print("   http://localhost:5000")
    print("\n✨ Features:")
    print("   • Hold currency steady for 2 seconds")
    print("   • System captures and shows result")
    print("   • Result clears after 5 seconds")
    print("   • Upload images for instant detection")
    print("\nPress Ctrl+C to stop\n")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


def preprocess_frame(frame):
    """Preprocess frame for model prediction"""
    # Resize
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize
    img = img.astype('float32') / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def detect_currency(frame):
    """Detect currency in frame and return predictions with strict validation"""
    if model is None:
        return None
    
    # Preprocess
    processed = preprocess_frame(frame)
    
    # Predict
    predictions = model.predict(processed, verbose=0)[0]
    
    # Sort predictions to find top 2
    pred_indices = np.argsort(predictions)[::-1]  # Descending order
    top1_idx = pred_indices[0]
    top2_idx = pred_indices[1] if len(pred_indices) > 1 else top1_idx
    
    top1_conf = predictions[top1_idx]
    top2_conf = predictions[top2_idx]
    
    predicted_class = class_names[top1_idx]
    
    # Calculate confidence gap between top 2 predictions
    confidence_gap = top1_conf - top2_conf
    
    # Check if it's not_currency or low confidence
    is_unknown = False
    display_class = predicted_class
    reason = ""
    
    # Rule 1: If predicted as not_currency
    if predicted_class == 'not_currency':
        is_unknown = True
        display_class = 'Unknown'
        reason = 'Not Currency'
    
    # Rule 2: If confidence is too low (uncertain)
    elif top1_conf < UNKNOWN_THRESHOLD:
        is_unknown = True
        display_class = 'Unknown'
        reason = f'Low Confidence ({top1_conf*100:.1f}%)'
    
    # Rule 3: If confidence gap is too small (can't decide between options)
    elif confidence_gap < MIN_CONFIDENCE_GAP:
        is_unknown = True
        display_class = 'Unknown'
        reason = f'Uncertain (Gap: {confidence_gap*100:.1f}%)'
    
    # Rule 4: If has not_currency class, check if it's getting significant votes
    elif has_not_currency and 'not_currency' in class_names:
        not_curr_idx = class_names.index('not_currency')
        not_curr_conf = predictions[not_curr_idx]
        
        # If not_currency is in top 2 predictions
        if not_curr_idx in [top1_idx, top2_idx]:
            is_unknown = True
            display_class = 'Unknown'
            reason = f'Possibly Not Currency ({not_curr_conf*100:.1f}%)'
        # Or if not_currency has significant confidence (>20%)
        elif not_curr_conf > 0.20:
            is_unknown = True
            display_class = 'Unknown'
            reason = f'Mixed Signal ({not_curr_conf*100:.1f}% not currency)'
    
    # Rule 5: Check for impossible predictions (all classes similar)
    std_dev = np.std(predictions)
    if std_dev < 0.15:  # All predictions are very similar
        is_unknown = True
        display_class = 'Unknown'
        reason = 'No Clear Match'
    
    return {
        'class': display_class,
        'class_idx': int(top1_idx),
        'confidence': float(top1_conf),
        'confidence_gap': float(confidence_gap),
        'is_unknown': is_unknown,
        'unknown_reason': reason,
        'original_class': predicted_class,
        'all_predictions': {class_names[i]: float(predictions[i]) for i in range(len(class_names))},
        'top2_gap': float(confidence_gap)
    }

def draw_detection_overlay(frame, detection):
    """Draw bounding box and info overlay on frame"""
    height, width = frame.shape[:2]
    
    if detection and detection['confidence'] >= CONFIDENCE_THRESHOLD and not detection.get('is_unknown', False):
        # Valid currency detection - GREEN box
        box_w = int(width * 0.7)
        box_h = int(height * 0.5)
        x1 = (width - box_w) // 2
        y1 = (height - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h
        
        color = (0, 255, 100)  # Green
        thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw corner markers
        corner_len = 40
        corner_thick = 5
        
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, corner_thick)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, corner_thick)
        
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, corner_thick)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, corner_thick)
        
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, corner_thick)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, corner_thick)
        
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, corner_thick)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, corner_thick)
        
        # Add glow effect
        glow_color = (0, 255, 100)
        cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), glow_color, 1)
        cv2.rectangle(frame, (x1-8, y1-8), (x2+8, y2+8), glow_color, 1)
        
        # Draw scanning line
        scan_y = int(y1 + (y2 - y1) * ((datetime.now().microsecond / 1000000) % 1))
        cv2.line(frame, (x1, scan_y), (x2, scan_y), (0, 255, 255), 2)
        
        # Draw label
        label = f"{detection['class']}: {detection['confidence']*100:.1f}%"
        gap_text = f"Gap: {detection['confidence_gap']*100:.0f}%"
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap_size = cv2.getTextSize(gap_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Main label background
        cv2.rectangle(frame, 
                     (x1, y1 - label_size[1] - 45),
                     (x1 + max(label_size[0], gap_size[0]) + 20, y1),
                     (0, 255, 100), -1)
        
        # Main text
        cv2.putText(frame, label, 
                   (x1 + 10, y1 - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        # Gap text (smaller, below)
        cv2.putText(frame, gap_text, 
                   (x1 + 10, y1 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Detection indicator
        cv2.circle(frame, (width - 30, 30), 15, (0, 255, 100), -1)
        cv2.putText(frame, "LIVE", (width - 80, 38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 2)
        
    elif detection and detection.get('is_unknown', False):
        # Unknown detection - YELLOW/ORANGE box
        box_w = int(width * 0.7)
        box_h = int(height * 0.5)
        x1 = (width - box_w) // 2
        y1 = (height - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h
        
        color = (0, 165, 255)  # Orange
        thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw corner markers
        corner_len = 40
        corner_thick = 5
        
        # All corners
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, corner_thick)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, corner_thick)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, corner_thick)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, corner_thick)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, corner_thick)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, corner_thick)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, corner_thick)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, corner_thick)
        
        # Draw label with reason
        label = detection['class']
        reason = detection.get('unknown_reason', '')
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        reason_size = cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Background
        bg_width = max(label_size[0], reason_size[0]) + 20
        cv2.rectangle(frame, 
                     (x1, y1 - label_size[1] - 45),
                     (x1 + bg_width, y1),
                     (0, 165, 255), -1)
        
        # Main text
        cv2.putText(frame, label, 
                   (x1 + 10, y1 - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Reason text
        if reason:
            cv2.putText(frame, reason, 
                       (x1 + 10, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Unknown indicator
        cv2.circle(frame, (width - 30, 30), 15, (0, 165, 255), -1)
        cv2.putText(frame, "?", (width - 37, 38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    else:
        # No detection - show guide frame
        box_w = int(width * 0.7)
        box_h = int(height * 0.5)
        x1 = (width - box_w) // 2
        y1 = (height - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h
        
        # Dashed rectangle guide
        dash_length = 20
        gap_length = 10
        color = (100, 100, 100)  # Gray
        
        # Top and bottom
        for x in range(x1, x2, dash_length + gap_length):
            cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, 2)
            cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, 2)
        
        # Left and right
        for y in range(y1, y2, dash_length + gap_length):
            cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, 2)
            cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, 2)
        
        # Instruction text
        text = "Position currency in frame"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = y2 + 40
        
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    return frame

def generate_frames():
    """Generate video frames with detection overlay"""
    global camera, detection_stats
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame_count += 1
        
        # Detect currency
        detection = detect_currency(frame)
        
        # Update stats (only for valid currency detections, not Unknown)
        if detection and detection['confidence'] >= CONFIDENCE_THRESHOLD and not detection.get('is_unknown', False):
            detection_stats['total_detections'] += 1
            # Running average of confidence
            prev_avg = detection_stats['avg_confidence']
            n = detection_stats['total_detections']
            detection_stats['avg_confidence'] = (prev_avg * (n-1) + detection['confidence']) / n
        
        # Draw overlay
        frame = draw_detection_overlay(frame, detection)
        
        # Add stats overlay (top-left)
        stats_text = [
            f"Detections: {detection_stats['total_detections']}",
            f"Avg Conf: {detection_stats['avg_confidence']*100:.1f}%"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 30
        
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

@app.route('/stats')
def get_stats():
    """Get detection statistics"""
    return jsonify(detection_stats)

@app.route('/capture', methods=['POST'])
def manual_capture():
    """Manually trigger a capture"""
    # This would require global frame storage
    return jsonify({'status': 'captured'})

@app.route('/settings', methods=['POST'])
def update_settings():
    """Update detection settings"""
    global CONFIDENCE_THRESHOLD
    
    data = request.json
    if 'threshold' in data:
        CONFIDENCE_THRESHOLD = float(data['threshold'])
    
    return jsonify({'status': 'updated'})

# @app.route('/predict_upload', methods=['POST'])
# def predict_upload():
#     """Handle uploaded image prediction"""
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400
    
#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No image selected'}), 400
    
#     try:
#         # Read image
#         img_bytes = file.read()
#         nparr = np.frombuffer(img_bytes, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         # Detect
#         detection = detect_currency(img)
        
#         if detection:
#             # Create predictions list
#             predictions = []
#             for class_name, conf in detection['all_predictions'].items():
#                 predictions.append({
#                     'class': class_name,
#                     'confidence': conf
#                 })
            
#             # Update stats if valid detection
#             if detection['confidence'] >= CONFIDENCE_THRESHOLD and not detection.get('is_unknown', False):
#                 detection_stats['total_detections'] += 1
#                 prev_avg = detection_stats['avg_confidence']
#                 n = detection_stats['total_detections']
#                 detection_stats['avg_confidence'] = (prev_avg * (n-1) + detection['confidence']) / n
            
#             return jsonify({
#                 'success': True,
#                 'predictions': predictions,
#                 'best_match': detection['class'],
#                 'confidence': detection['confidence']
#             })
#         else:
#             return jsonify({'error': 'Detection failed'}), 500
            
#     except Exception as e:
#         print(f"Upload prediction error: {e}")
#         return jsonify({'error': str(e)}), 500

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    """Handle uploaded image prediction"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect
        detection = detect_currency(img)
        
        if detection:
            # Create predictions list
            predictions = []
            for class_name, conf in detection['all_predictions'].items():
                predictions.append({
                    'class': class_name,
                    'confidence': conf
                })
            
            # Update stats if valid detection
            if detection['confidence'] >= CONFIDENCE_THRESHOLD and not detection.get('is_unknown', False):
                detection_stats['total_detections'] += 1
                prev_avg = detection_stats['avg_confidence']
                n = detection_stats['total_detections']
                detection_stats['avg_confidence'] = (prev_avg * (n-1) + detection['confidence']) / n
            
            # Encode image to base64 for display in browser
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'best_match': detection['class'],
                'confidence': detection['confidence'],
                'is_unknown': detection.get('is_unknown', False),
                'image': img_base64  # Added missing image field
            })
        else:
            return jsonify({'error': 'Detection failed'}), 500
            
    except Exception as e:
        print(f"Upload prediction error: {e}")
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    print("="*60)
    print("CURRENCY VISION - WEB SERVER")
    print("="*60)
    
    # Load model
    if not load_model_and_classes():
        print("\n❌ Please train the model first: python train.py")
        exit(1)
    
    print("\n" + "="*60)
    print("Starting web server...")
    print("="*60)
    print("\n🌐 Open your browser and go to:")
    print("   http://localhost:5000")
    print("\n✨ Features:")
    print("   • Live webcam detection with green bounding box")
    print("   • Real-time confidence scores")
    print("   • Automatic image capture for retraining")
    print("   • Detection statistics")
    print("\n📁 Captured images saved to: captured_images/")
    print("\nPress Ctrl+C to stop\n")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
