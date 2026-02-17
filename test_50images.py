"""
Currency Recognition System - Random Testing with Debug
"""

import os
import random
import tensorflow as tf
import numpy as np
import cv2

print("="*60)
print("CURRENCY TESTING - DEBUG MODE")
print("="*60)

# Check current directory
current_dir = os.getcwd()
print(f"\n📁 Current directory: {current_dir}")

# Check if dataset folder exists
dataset_path = 'dataset'
if os.path.exists(dataset_path):
    print(f"✅ Found dataset folder: {dataset_path}")
else:
    print(f"❌ Dataset folder not found: {dataset_path}")
    print("   Trying absolute path...")
    dataset_path = os.path.join(current_dir, 'dataset')
    if os.path.exists(dataset_path):
        print(f"✅ Found dataset folder: {dataset_path}")
    else:
        print(f"❌ Still not found. Please check folder name.")

# List contents of current directory
print(f"\n📂 Contents of current directory:")
for item in os.listdir('.'):
    if os.path.isdir(item):
        print(f"   📁 {item}/")
    else:
        print(f"   📄 {item}")

# Load model
print(f"\n🤖 Loading model...")
model_path = 'models/currency_model.h5'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded successfully")
else:
    print(f"❌ Model not found at: {model_path}")

# Load class names
class_names_path = 'models/class_names.txt'
if os.path.exists(class_names_path):
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"✅ Class names loaded: {class_names}")
else:
    print(f"❌ class_names.txt not found")
    class_names = []

# Get all images from dataset
print(f"\n🔍 Searching for images in dataset...")

all_images = []
if os.path.exists('dataset'):
    for class_name in class_names:
        class_dir = os.path.join('dataset', class_name)
        print(f"\n   Checking class: {class_name}")
        print(f"   Path: {class_dir}")
        
        if os.path.exists(class_dir):
            print(f"   ✅ Class folder exists")
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"   Found {len(images)} images")
            
            for img in images:
                all_images.append(os.path.join(class_dir, img))
        else:
            print(f"   ❌ Class folder NOT found")

print(f"\n📊 TOTAL IMAGES FOUND: {len(all_images)}")

if len(all_images) == 0:
    print("\n❌ No images found! Please check:")
    print("   1. Your dataset folder structure should be:")
    print("      dataset/")
    print("      ├── class1/")
    print("      │   ├── image1.jpg")
    print("      │   └── image2.jpg")
    print("      ├── class2/")
    print("      │   └── ...")
    print("   2. Image formats should be .jpg, .jpeg, or .png")
    exit()

# Test 50 random images
print(f"\n🧪 Testing 50 random images...")

random.shuffle(all_images)
test_images = all_images[:50]

correct = 0
total = 0

for i, img_path in enumerate(test_images, 1):
    try:
        # Get actual class from folder name
        actual_class = os.path.basename(os.path.dirname(img_path))
        
        # Load and predict
        img = cv2.imread(img_path)
        if img is None:
            print(f"   {i:2d}. ❌ Cannot read image: {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)) / 255.0
        img = np.expand_dims(img, axis=0)
        
        pred = model.predict(img, verbose=0)[0]
        confidence = np.max(pred)
        predicted = class_names[np.argmax(pred)]
        
        # Check if correct (predicted matches actual AND confidence >= 90%)
        is_correct = (predicted == actual_class and confidence >= 0.90)
        if is_correct:
            correct += 1
        total += 1
        
        status = "✅" if is_correct else "❌"
        print(f"   {i:2d}. {status} Actual: {actual_class:15s} Predicted: {predicted:15s} Conf: {confidence*100:5.1f}%")
        
    except Exception as e:
        print(f"   {i:2d}. ❌ Error: {img_path} - {str(e)}")

# Results
print("\n" + "="*60)
print("TEST RESULTS")
print("="*60)
print(f"Total Images Tested: {total}")
print(f"Correct Detections:  {correct}/{total}")
if total > 0:
    print(f"Accuracy:           {(correct/total)*100:.1f}%")
else:
    print(f"Accuracy:           0%")
print("="*60)