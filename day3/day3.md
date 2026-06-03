# Day 3
## Single Neuron Example (Perceptron)
```py
import numpy as np

# Inputs
inputs = np.array([1, 2, 3])

# Weights
weights = np.array([0.2, 0.8, -0.5])

# Bias
bias = 2

# Output
output = np.dot(inputs, weights) + bias

print("Neuron Output:", output)
```

## 2.Activation Function Example

 
```py
import numpy as np
def sigmoid(x):
   return 1 / (1 + np.exp(-x))

x = 2
print("Sigmoid Output:", sigmoid(x))
```
## 3.XOR Problem Using Neural Network

```py
from sklearn.neural_network import MLPClassifier
# XOR Dataset
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,1,1,0]
# Neural Network
model = MLPClassifier(hidden_layer_sizes=(4,),max_iter=5000,random_state=42)
model.fit(X, y)
print("Predictions:")
print(model.predict(X))
```

## 4. Visualizing a Neural Network

 ```py
 import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))

# Input Layer
plt.scatter([1,1],[1,2],s=500)

# Hidden Layer
plt.scatter([3,3,3],[0.5,1.5,2.5],s=500)

# Output Layer
plt.scatter([5],[1.5],s=500)
plt.title("Simple Neural Network")
plt.axis('off')
plt.show()

 ```

 ## Problem 2
 ### Step 1: Check GPU Availability

 
```py
import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

print("GPU Available:", tf.config.list_physical_devices('GPU'))
```
 
### Step 2: Neural Network Program

```py
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
 
# Load Fashion-MNIST Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
 
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
 
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
 
# Normalize Pixel Values
X_train = X_train / 255.0
X_test = X_test / 255.0
 
# Build Neural Network
model = tf.keras.Sequential([
   tf.keras.layers.Flatten(input_shape=(28,28)),
 
   tf.keras.layers.Dense(256, activation='relu'),
   tf.keras.layers.Dropout(0.2),
 
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.Dropout(0.2),
 
   tf.keras.layers.Dense(64, activation='relu'),
 
   tf.keras.layers.Dense(10, activation='softmax')
])
 
# Compile Model
model.compile(
   optimizer='adam',
   loss='sparse_categorical_crossentropy',
   metrics=['accuracy']
)
 # Train Model on GPU
history = model.fit(
   X_train,
   y_train,
   epochs=10,
   batch_size=128,
   validation_split=0.2
)
 
# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
 
print("\nTest Accuracy:", test_accuracy)
 
# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
 
# Confusion Matrix
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_classes))
 
# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred_classes))
```
### Step 3: Display Sample Predictions

```py
class_names = [
   'T-shirt','Trouser','Pullover','Dress',
   'Coat','Sandal','Shirt','Sneaker',
   'Bag','Ankle boot'
]
 
plt.figure(figsize=(10,5))
 
for i in range(10):
   plt.subplot(2,5,i+1)
   plt.imshow(X_test[i], cmap='gray')
   plt.title(class_names[y_pred_classes[i]])
   plt.axis('off')
 
plt.show()
```

## intel image classification

```py
# ==========================================
# Intel Image Classification using CNN
# Google Colab Complete Code
# ==========================================

# Install required packages (if needed)
# !pip install -q seaborn

# ==========================================
# Import Libraries
# ==========================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from google.colab import files
import zipfile
import os

# ==========================================
# Upload Dataset ZIP File
# ==========================================

print("Upload Intel Dataset ZIP file")
uploaded = files.upload()

zip_file = list(uploaded.keys())[0]

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall('/content/')

print("Dataset Extracted Successfully!")

# ==========================================
# Dataset Paths
# ==========================================

train_dir = "/content/seg_train"
test_dir = "/content/seg_test"

# Verify paths
print("Train Exists:", os.path.exists(train_dir))
print("Test Exists:", os.path.exists(test_dir))

# ==========================================
# Parameters
# ==========================================

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# ==========================================
# Load Dataset
# ==========================================

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='int'
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode='int'
)

# Class Names
class_names = train_dataset.class_names

print("\nClasses:")
print(class_names)

# ==========================================
# Normalize Images
# ==========================================

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

test_dataset = test_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

# Improve Performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)

# ==========================================
# Build CNN Model
# ==========================================

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(150,150,3)
    ),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(
        128,
        (3,3),
        activation='relu'
    ),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(
        256,
        (3,3),
        activation='relu'
    ),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        256,
        activation='relu'
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        6,
        activation='softmax'
    )
])

# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# Model Summary
# ==========================================

model.summary()

# ==========================================
# Train Model
# ==========================================

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset
)

# ==========================================
# Evaluate Model
# ==========================================

test_loss, test_accuracy = model.evaluate(test_dataset)

print("\n========================")
print("Test Accuracy:", test_accuracy)
print("========================")

# ==========================================
# Plot Accuracy & Loss
# ==========================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# ==========================================
# Predictions
# ==========================================

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# ==========================================
# Display Sample Predictions
# ==========================================

plt.figure(figsize=(15,8))

for images, labels in test_dataset.take(1):

    predictions = model.predict(
        images,
        verbose=0
    )

    pred_classes = np.argmax(
        predictions,
        axis=1
    )

    for i in range(10):

        plt.subplot(2,5,i+1)

        plt.imshow(images[i].numpy())

        actual = class_names[labels[i]]
        predicted = class_names[pred_classes[i]]

        plt.title(
            f"A:{actual}\nP:{predicted}",
            fontsize=9
        )

        plt.axis("off")

plt.tight_layout()
plt.show()

# ==========================================
# Save Model
# ==========================================

model.save("intel_image_classifier.h5")

print("\nModel saved as intel_image_classifier.h5")
```