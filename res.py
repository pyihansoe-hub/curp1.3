"""
real_test_fixed.py - Get ACTUAL results with Unicode path handling
"""

import os
import random
import tensorflow as tf
import cv2
import numpy as np
from collections import defaultdict
import sys

# Force UTF-8 encoding for console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*60)
print("CURRENCY RECOGNITION - REAL TEST RESULTS")
print("="*60)

# Load class names
print("\n📂 Loading class names...")
with open('models/class_names.txt', 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f.readlines()]
print(f"✅ Classes: {class_names}")

# Load model
print("\n📂 Loading model...")
model = tf.keras.models.load_model('models/currency_model.h5')
print("✅ Model loaded successfully")

# Test parameters
CONFIDENCE_THRESHOLD = 0.9
TEST_IMAGES_PER_CLASS = 10  # Test 10 images per class

dataset_path = 'dataset'
print(f"\n📁 Dataset path: {os.path.abspath(dataset_path)}")

# Store results
results = {}
total_tested = 0
total_correct = 0
all_confidences = []

print("\n" + "="*60)
print("TESTING EACH CLASS")
print("="*60)

for class_name in class_names:
    print(f"\n🔍 Testing: {class_name}")
    
    # Use raw string to handle Unicode
    class_dir = os.path.join(dataset_path, class_name)
    
    if not os.path.exists(class_dir):
        print(f"   ❌ Folder not found: {class_dir}")
        continue
    
    # Get all image files
    try:
        all_files = os.listdir(class_dir)
    except Exception as e:
        print(f"   ❌ Cannot list directory: {e}")
        continue
    
    # Filter images
    image_files = []
    for f in all_files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(f)
    
    print(f"   📸 Found {len(image_files)} images")
    
    if len(image_files) == 0:
        continue
    
    # Select random images
    test_count = min(TEST_IMAGES_PER_CLASS, len(image_files))
    test_images = random.sample(image_files, test_count)
    
    class_correct = 0
    class_confidences = []
    
    for img_file in test_images:
        img_path = os.path.join(class_dir, img_file)
        
        try:
            # Read image with imdecode to handle Unicode better
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"   ⚠️  Cannot read: {img_file}")
                continue
            
            # Preprocess
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)) / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Predict
            pred = model.predict(img, verbose=0)[0]
            confidence = np.max(pred)
            predicted = class_names[np.argmax(pred)]
            
            # Check if correct
            is_correct = (predicted == class_name and confidence >= CONFIDENCE_THRESHOLD)
            
            if is_correct:
                class_correct += 1
                total_correct += 1
            
            class_confidences.append(confidence)
            total_tested += 1
            
            status = "✅" if is_correct else "❌"
            print(f"   {status} {img_file[:30]:<30} Pred: {predicted:<15} Conf: {confidence:.1%}")
            
        except Exception as e:
            print(f"   ❌ Error: {img_file} - {e}")
    
    # Class summary
    if test_count > 0:
        class_accuracy = (class_correct / test_count) * 100
        class_avg_conf = np.mean(class_confidences) * 100 if class_confidences else 0
        print(f"\n   📊 {class_name} Results:")
        print(f"      Accuracy: {class_correct}/{test_count} = {class_accuracy:.1f}%")
        print(f"      Avg Confidence: {class_avg_conf:.1f}%")
        
        results[class_name] = {
            'correct': class_correct,
            'total': test_count,
            'accuracy': class_accuracy,
            'avg_confidence': class_avg_conf
        }

# Overall results
print("\n" + "="*60)
print("OVERALL RESULTS")
print("="*60)

if total_tested > 0:
    overall_accuracy = (total_correct / total_tested) * 100
    overall_avg_conf = np.mean(all_confidences) * 100 if all_confidences else 0
    
    print(f"\n📊 Total images tested: {total_tested}")
    print(f"📊 Total correct: {total_correct}")
    print(f"📊 Overall accuracy: {overall_accuracy:.1f}%")
    print(f"📊 Average confidence: {overall_avg_conf:.1f}%")
    
    print("\n📊 Per-class breakdown:")
    print("-" * 60)
    print(f"{'Class':<20} {'Correct':<10} {'Accuracy':<12} {'Avg Conf':<10}")
    print("-" * 60)
    
    for class_name, res in results.items():
        print(f"{class_name:<20} {res['correct']:>2}/{res['total']:<3} {res['accuracy']:>6.1f}%{'':<4} {res['avg_confidence']:>6.1f}%")
    
    print("-" * 60)
    
    # Summary paragraph for your document
    print("\n" + "="*60)
    print("📝 SUMMARY FOR YOUR DOCUMENT")
    print("="*60)
    print(f"""
The system achieved an overall detection accuracy of {overall_accuracy:.1f} percent across {total_tested} test images.
The average confidence score for correct detections was {overall_avg_conf:.1f} percent.

Per-class performance:
""")
    for class_name, res in results.items():
        print(f"• {class_name}: {res['accuracy']:.1f}% accuracy with {res['avg_confidence']:.1f}% average confidence")
    
else:
    print("\n❌ No images were tested successfully!")
    print("\nTroubleshooting tips:")
    print("1. Make sure your dataset folder structure is:")
    print("   dataset/")
    for name in class_names:
        print(f"   ├── {name}/")
        print(f"   │   ├── image1.jpg")
    print("2. Try running Python with UTF-8 encoding:")
    print("   $env:PYTHONUTF8=1; python real_test_fixed.py")

print("\n" + "="*60)