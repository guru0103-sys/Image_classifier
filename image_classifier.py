import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np
import os
import shutil
import pathlib

# ==========================================
# STEP 0: CREATE DUMMY DATA (FOR DEMO ONLY)
# ==========================================
# Run this once to create a fake dataset structure so the code works immediately.
# Delete this block when using your real data.

BASE_DIR = r"C:\Users\DELL\Desktop\Guru\Cdes\my_Dataset"
# ==========================================
# STEP 1: CONFIGURATION & PREPARATION
# ==========================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
DATA_DIR = pathlib.Path(BASE_DIR)

# Get list of all file paths
all_image_paths = list(DATA_DIR.glob('**/*.jpg'))
all_image_paths = [str(path) for path in all_image_paths]
np.random.shuffle(all_image_paths) # Shuffle data

# get class names from folder names
unique_labels = set()
for path in all_image_paths:
    # getting the parent folder name of the image
    parent_folder = pathlib.Path(path).parent.name
    unique_labels.add(parent_folder)

class_names = sorted(list(unique_labels))
print(f"Classes found: {class_names}")
class_indices = dict(zip(class_names, range(len(class_names))))

print(f"Classes found: {class_names}")

# ==========================================
# STEP 2: CUSTOM DATA LOADER
# ==========================================
# This function teaches TensorFlow how to read your specific filenames
# It expects filenames to contain 'ai' or 'real' (e.g., 'cat_ai_01.jpg')

def process_path(file_path):
    # 1. Load the Image
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize to 0-1 range

    # 2. Extract Labels
    # Split path to get folder name (Subject Label)
    parts = tf.strings.split(file_path, os.path.sep)
    folder_name = parts[-2]
    
    # Map folder name to integer ID (e.g., cat=0, dog=1)
    subject_label = -1
    for name, index in class_indices.items():
        if folder_name == name:
            subject_label = index
    
    # 3. Extract Source Label (AI vs Real) from filename
    filename = parts[-1]
    # Check if 'ai' is in the filename. Returns 1.0 if AI, 0.0 if Real
    # Note: ensure your files are named correctly!
    is_ai = tf.strings.regex_full_match(filename, ".*ai.*") 
    source_label = tf.cast(is_ai, tf.float32)

    # Convert subject label to One-Hot encoding
    subject_label_one_hot = tf.one_hot(subject_label, depth=len(class_names))

    # RETURN: (Input, {Output1, Output2})
    return img, {'subject_output': subject_label_one_hot, 'source_output': source_label}

# Create TensorFlow Dataset pipeline
list_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

# Batch and pre-fetch for performance
train_ds = labeled_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==========================================
# STEP 3: BUILD THE MULTI-HEAD MODEL
# ==========================================

def build_model(num_classes):
    input_layer = layers.Input(shape=(224, 224, 3), name='image_input')

    # Backbone (MobileNetV2)
    base_model = applications.MobileNetV2(include_top=False, weights='imagenet', input_tensor=input_layer)
    base_model.trainable = False # Freeze for initial training

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)

    # HEAD 1: Subject Classifier (Softmax for multi-class)
    subj_branch = layers.Dense(64, activation='relu')(x)
    subj_output = layers.Dense(num_classes, activation='softmax', name='subject_output')(subj_branch)

    # HEAD 2: Source Detector (Sigmoid for binary Real/AI)
    src_branch = layers.Dense(32, activation='relu')(x)
    src_branch = layers.Dropout(0.3)(src_branch)
    src_output = layers.Dense(1, activation='sigmoid', name='source_output')(src_branch)

    model = models.Model(inputs=input_layer, outputs=[subj_output, src_output])
    return model

model = build_model(len(class_names))

model.compile(
    optimizer='adam',
    loss={
        'subject_output': 'categorical_crossentropy',
        'source_output': 'binary_crossentropy'
    },
    loss_weights={'subject_output': 1.0, 'source_output': 1.5},
    # CHANGE IS HERE: We specify accuracy for BOTH outputs specifically
    metrics={
        'subject_output': 'accuracy',
        'source_output': 'accuracy'
    }
)


# ==========================================
# STEP 4: TRAIN
# ==========================================

print("Starting training...")
history = model.fit(train_ds, epochs=5) # Increase epochs for real data

# ==========================================
# STEP 5: TESTING / PREDICTION
# ==========================================

print("\n--- Testing on a sample image ---")
# Pick one random image from our dataset to test
test_path = all_image_paths[0] 
print(f"Testing file: {test_path}")

# Preprocess single image
img = tf.io.read_file(test_path)
img = tf.io.decode_jpeg(img, channels=3)
img = tf.image.resize(img, IMG_SIZE) / 255.0
img = tf.expand_dims(img, axis=0) # Add batch dimension

# Predict
predictions = model.predict(img)
pred_subject = predictions[0] # First output head
pred_source = predictions[1]  # Second output head

# Decode Subject
subj_idx = np.argmax(pred_subject)
subj_confidence = np.max(pred_subject)
predicted_class = class_names[subj_idx]

# Decode Source
ai_probability = pred_source[0][0]
source_status = "AI Generated" if ai_probability > 0.5 else "Real Image"

print(f"\nPREDICTION RESULTS:")
print(f"1. Object: {predicted_class} ({subj_confidence*100:.2f}%)")
print(f"2. Source: {source_status} (AI Prob: {ai_probability:.4f})")