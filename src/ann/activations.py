"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np


class ReLU:
    def forward (self,Z):
        self.Z = Z
        return np.maximum(0,Z)
    
    def backward(self, dA):
        dZ = dA.copy()
        dZ[self.Z <=0] =0
        return dZ
    
class Sigmoid:
    def forward(self,Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A
    
    def backward(self, dA):
        return dA * self.A * (1-self.A)
    
class Tanh:
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A
    
    def backward(self, dA):
        return dA * (1- self.A**2)
    
class Softmax:
    def forward(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.probs = expZ / np.sum(expZ, axis=1, keepdims=True)
        return self.probs

    def backward(self, y):
        m = y.shape[0]
        return (self.probs - y) / m

if __name__ == "__main__":

    import numpy as np

    Z = np.array([[-1.0, 2.0, -3.0],
                [4.0, -5.0, 6.0]])

    print("Input Z:")
    print(Z)
    print("\n")

    # ReLU Test
    relu = ReLU()
    A_relu = relu.forward(Z)
    print("ReLU Forward:")
    print(A_relu)

    dA = np.ones_like(Z)
    print("ReLU Backward:")
    print(relu.backward(dA))
    print("\n")

    # Sigmoid Test
    sigmoid = Sigmoid()
    A_sig = sigmoid.forward(Z)
    print("Sigmoid Forward:")
    print(A_sig)

    print("Sigmoid Backward:")
    print(sigmoid.backward(dA))
    print("\n")

    # Tanh Test
    tanh = Tanh()
    A_tanh = tanh.forward(Z)
    print("Tanh Forward:")
    print(A_tanh)

    print("Tanh Backward:")
    print(tanh.backward(dA))
    print("\n")

    # Softmax Test
    softmax = Softmax()
    A_soft = softmax.forward(Z)
    print("Softmax Forward:")
    print(A_soft)

    # Fake one-hot labels for backward test
    y = np.array([[1, 0, 0],
                [0, 0, 1]])

    print("Softmax Backward:")
    print(softmax.backward(y))  