import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import numpy as np


# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 100
DATA_DIR = 'dataset/'  
MODEL_SAVE_PATH = 'models/currency_model.h5'

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("=" * 70)
print("ADVANCED CURRENCY RECOGNITION TRAINING")
print("=" * 70)

# Check dataset
print("\n[1/6] Checking dataset...")
if not os.path.exists(DATA_DIR):
    print(f"❌ Error: Dataset folder '{DATA_DIR}' not found!")
    print("Please create it and add your currency images.")
    exit(1)

folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
print(f"✓ Found {len(folders)} classes: {folders}")

# Advanced data augmentation
print("\n[2/6] Setting up advanced data augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,           # Rotate up to 25 degrees
    width_shift_range=0.2,       # Shift left/right 20%
    height_shift_range=0.2,      # Shift up/down 20%
    shear_range=0.15,            # Shear transformation
    zoom_range=0.25,             # Zoom in/out 25%
    horizontal_flip=True,        # Flip horizontally
    brightness_range=[0.7, 1.3], # Brightness variation
    channel_shift_range=20,      # Color variation
    fill_mode='nearest',
    validation_split=0.2         # 80% train, 20% validation
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Load data
print("\n[3/6] Loading and preparing data...")

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)

print(f"\n✓ Training images: {train_generator.samples}")
print(f"✓ Validation images: {validation_generator.samples}")
print(f"✓ Classes: {class_names}")
print(f"✓ Images per class:")
for class_name in class_names:
    class_dir = os.path.join(DATA_DIR, class_name)
    count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"   - {class_name}: {count} images")

# Build enhanced model
print("\n[4/6] Building enhanced model architecture...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'  # Pre-trained on ImageNet
)

# Freeze base model initially
base_model.trainable = False

# Build custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n✓ Model built successfully")
print(f"✓ Total parameters: {model.count_params():,}")

# Setup callbacks
print("\n[5/6] Configuring training callbacks...")

callbacks = [
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
]

# Training Phase 1 - Train only top layers
print("\n[6/6] Starting training...")
print("=" * 70)
print("PHASE 1: Training classification head (base frozen)")
print("=" * 70)

history1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

# Phase 2: Fine-tune with unfrozen layers
print("\n" + "=" * 70)
print("PHASE 2: Fine-tuning with unfrozen top layers")
print("=" * 70)

base_model.trainable = True

# Freeze early layers, train only last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n✓ Unfrozen top layers for fine-tuning")

history2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=70,
    callbacks=callbacks,
    initial_epoch=30,
    verbose=1
)

# Save results
print("\n" + "=" * 70)
print("Saving training results...")
print("=" * 70)

# Combine histories
history = {
    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss': history1.history['loss'] + history2.history['loss'],
    'val_loss': history1.history['val_loss'] + history2.history['val_loss']
}

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy
axes[0].plot(history['accuracy'], label='Training', linewidth=2)
axes[0].plot(history['val_accuracy'], label='Validation', linewidth=2)
axes[0].axvline(x=30, color='red', linestyle='--', label='Fine-tuning Start')
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history['loss'], label='Training', linewidth=2)
axes[1].plot(history['val_loss'], label='Validation', linewidth=2)
axes[1].axvline(x=30, color='red', linestyle='--', label='Fine-tuning Start')
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
print("✓ Training plots saved to: results/training_history.png")

# Save class names
with open('models/class_names.txt', 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")
print("✓ Class names saved to: models/class_names.txt")

# Final summary
print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"\n📊 Results:")
print(f"   • Best Model: {MODEL_SAVE_PATH}")
print(f"   • Training Accuracy: {max(history['accuracy']):.2%}")
print(f"   • Validation Accuracy: {max(history['val_accuracy']):.2%}")

print(f"\n📁 Class Distribution:")
for i, name in enumerate(class_names):
    class_dir = os.path.join(DATA_DIR, name)
    count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"   {i}: {name:15s} - {count:3d} images")


print(f"   Run web app: python app.py")
print(f"   Add more data to dataset/ for better accuracy")

print("\n" + "=" * 70)

