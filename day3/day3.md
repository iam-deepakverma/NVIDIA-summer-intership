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