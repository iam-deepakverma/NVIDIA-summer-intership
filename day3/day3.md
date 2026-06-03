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