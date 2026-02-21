
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
import sys

# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)
app.secret_key = 'currency_detection_secret_key'
CORS(app)

# Configuration
MODEL_PATH = 'models/currency_model.h5'
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.97
UNKNOWN_THRESHOLD = 0.95
CAPTURE_DIR = 'captured_images'
AUTO_CAPTURE_ENABLED = True
STABILIZATION_SECONDS = 2

# Global variables
model = None
class_names = []
has_not_currency = False
camera = None
detection_stats = {
    'total_detections': 0,
    'total_confidence_sum': 0,  # FIXED: Track sum for proper average
    'captures_saved': 0
}

# Detection tracking
detection_start_time = None
current_detection_class = None
current_detection_confidence = 0
stable_detection_frames = 0
latest_result = None
latest_result_image = None

# Create directories
os.makedirs(CAPTURE_DIR, exist_ok=True)

def load_model_and_classes():
    """Load the trained model and class names"""
    global model, class_names, has_not_currency
    
    print("Loading model...")
    if os.path.exists(MODEL_PATH):
        try:
            # FIXED: Load with custom objects if needed
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print("❌ Model not found at:", MODEL_PATH)
        print("Available models:")
        if os.path.exists('models'):
            for f in os.listdir('models'):
                if f.endswith('.h5') or f.endswith('.keras'):
                    print(f"  - {f}")
        return False
    
    # Load class names
    class_names_path = 'models/class_names.txt'
    if os.path.exists(class_names_path):
        try:
            with open(class_names_path, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]
            print(f"✓ Class names loaded: {class_names}")
            
            # FIXED: Create capture directories for each class
            for class_name in class_names:
                os.makedirs(os.path.join(CAPTURE_DIR, class_name), exist_ok=True)
            
            has_not_currency = 'not_currency' in class_names
        except Exception as e:
            print(f"❌ Error loading class names: {e}")
            return False
    else:
        print("⚠️ class_names.txt not found")
        # FIXED: Don't use default classes - require real class names
        return False
    
    return True

def preprocess_frame(frame):
    """Preprocess frame for model prediction"""
    try:
        # FIXED: Ensure correct color conversion
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return None

def detect_currency(frame):
    """Detect currency in frame and return predictions"""
    if model is None or frame is None:
        return None
    
    try:
        processed = preprocess_frame(frame)
        if processed is None:
            return None
        
        predictions = model.predict(processed, verbose=0)[0]
        
        max_idx = np.argmax(predictions)
        max_conf = float(predictions[max_idx])
        predicted_class = class_names[max_idx] if max_idx < len(class_names) else 'unknown'
        
        # Check if confidence is below threshold
        is_unknown = max_conf < CONFIDENCE_THRESHOLD or predicted_class == 'not_currency'
        
        # FIXED: Return all predictions with proper class names
        all_preds = {}
        for i, name in enumerate(class_names):
            if i < len(predictions):
                all_preds[name] = float(predictions[i])
        
        return {
            'class': 'Unknown' if is_unknown else predicted_class,
            'original_class': predicted_class,
            'confidence': max_conf,
            'is_unknown': is_unknown,
            'all_predictions': all_preds
        }
    except Exception as e:
        print(f"Error in detect_currency: {e}")
        return None

def draw_detection_overlay(frame, detection):
    """Draw bounding box overlay on frame"""
    if frame is None:
        return frame
    
    height, width = frame.shape[:2]
    
    # Guide box dimensions
    box_w = int(width * 0.7)
    box_h = int(height * 0.5)
    x1 = (width - box_w) // 2
    y1 = (height - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h
    
    try:
        if detection and not detection.get('is_unknown', False) and detection['confidence'] >= CONFIDENCE_THRESHOLD:
            # Valid currency - GREEN box
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
            
            # Label
            label = f"{detection['class']}: {detection['confidence']*100:.1f}%"
            cv2.putText(frame, label, (x1 + 10, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        elif detection:
            # Unknown - ORANGE box
            color = (0, 165, 255)
            thickness = 3
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, "Unknown", (x1 + 10, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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
        
        # Stats at top left
        cv2.putText(frame, f"Detected: {detection_stats['total_detections']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    except Exception as e:
        print(f"Error drawing overlay: {e}")
    
    return frame

def save_captured_image(frame, detection):
    """Save captured frame"""
    try:
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
    except Exception as e:
        print(f"Error saving image: {e}")
    
    return False

def generate_frames():
    """Generate video frames with detection overlay"""
    global camera, detection_stats, detection_start_time, current_detection_class
    global current_detection_confidence, stable_detection_frames, latest_result, latest_result_image
    
    # FIXED: Proper camera initialization with error handling
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("❌ Could not open camera")
        return
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    required_stable_frames = int(STABILIZATION_SECONDS * 10)  # ~10 FPS for stability
    
    while True:
        try:
            success, frame = camera.read()
            if not success:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Detect currency every frame
            detection = detect_currency(frame)
            
            # FIXED: Improved stabilization logic
            if detection and not detection.get('is_unknown', False) and detection['confidence'] >= CONFIDENCE_THRESHOLD:
                if current_detection_class == detection['original_class']:
                    stable_detection_frames += 1
                    
                    # FIXED: Use frame count instead of time for more reliable detection
                    if stable_detection_frames >= required_stable_frames and latest_result is None:
                        latest_result = detection
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        latest_result_image = base64.b64encode(buffer).decode('utf-8')
                        
                        # Update stats
                        detection_stats['total_detections'] += 1
                        detection_stats['total_confidence_sum'] += detection['confidence']
                        
                        # Save for retraining
                        save_captured_image(frame, detection)
                        
                        print(f"✅ Detected: {detection['class']} with {detection['confidence']*100:.1f}% confidence")
                else:
                    # New detection
                    current_detection_class = detection['original_class']
                    current_detection_confidence = detection['confidence']
                    stable_detection_frames = 1
            else:
                # No valid detection
                current_detection_class = None
                current_detection_confidence = 0
                stable_detection_frames = 0
            
            # Draw overlay
            frame = draw_detection_overlay(frame, detection)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            break
    
    # FIXED: Proper camera release
    if camera:
        camera.release()

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
    global latest_result, latest_result_image, detection_start_time
    global current_detection_class, stable_detection_frames
    
    latest_result = None
    latest_result_image = None
    detection_start_time = None
    current_detection_class = None
    stable_detection_frames = 0
    
    return jsonify({'status': 'cleared'})

@app.route('/stats')
def get_stats():
    """Get detection statistics"""
    avg_confidence = 0
    if detection_stats['total_detections'] > 0:
        avg_confidence = (detection_stats['total_confidence_sum'] / 
                         detection_stats['total_detections']) * 100
    
    return jsonify({
        'total_detections': detection_stats['total_detections'],
        'avg_confidence': avg_confidence,
        'captures_saved': detection_stats['captures_saved']
    })

@app.route('/settings', methods=['POST'])
def update_settings():
    """Update detection settings"""
    global CONFIDENCE_THRESHOLD
    
    data = request.json
    if 'threshold' in data:
        try:
            value = float(data['threshold'])
            if 0.5 <= value <= 1.0:
                CONFIDENCE_THRESHOLD = value
                return jsonify({'status': 'updated', 'threshold': value})
            else:
                return jsonify({'error': 'Threshold must be between 0.5 and 1.0'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid threshold value'}), 400
    
    return jsonify({'status': 'updated'})

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    """Handle uploaded image prediction"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # FIXED: Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return jsonify({'error': 'File type not allowed. Please upload JPG, JPEG, or PNG'}), 400
    
    try:
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Convert to base64 for response
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Detect currency
        detection = detect_currency(img)
        
        if detection:
            # Format predictions for frontend
            predictions = []
            for class_name, conf in detection['all_predictions'].items():
                predictions.append({
                    'class': class_name,
                    'confidence': conf
                })
            
            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'best_match': predictions[0]['class'] if predictions else 'Unknown',
                'confidence': predictions[0]['confidence'] if predictions else 0,
                'is_unknown': detection['is_unknown'],
                'image': img_base64
            })
        else:
            return jsonify({'error': 'Detection failed'}), 500
            
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("CURRENCY DETECTION SYSTEM - FIXED VERSION")
    print("="*60)
    
    if not load_model_and_classes():
        print("\n❌ Failed to load model. Please check:")
        print("   1. Model exists at: models/currency_model.h5")
        print("   2. Class names exist at: models/class_names.txt")
        print("   3. Run training first: python train.py")
        exit(1)
    
    print(f"\n🌐 Server running at: http://localhost:5001")
    print(f"📸 Hold currency for {STABILIZATION_SECONDS} seconds to detect")
    print(f"✅ Shows only if confidence >= {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"📁 Captured images saved to: {CAPTURE_DIR}/")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)