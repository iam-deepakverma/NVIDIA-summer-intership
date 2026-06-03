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