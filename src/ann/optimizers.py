"""
Optimization Algorithms
Implements: SGD, Momentum, RMSProp, NAG etc.
"""

import numpy as np

class SGD:
    
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        
    def update(self, layer):
        layer.W -= self.lr * layer.grad_W
        layer.b -= self.lr * layer.grad_b
        
class Momentum:
    
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.velocities = {}
        
    def update(self, layer):
        
        if layer not in self.velocities:
            self.velocities[layer] = {
                "vW": np.zeros_like(layer.W),
                "vB": np.zeros_like(layer.b)
            }
        
        vW = self.velocities[layer]["vW"]
        vB = self.velocities[layer]["vB"]
        
        vW = self.beta * vW + (1 - self.beta) * layer.grad_W
        vB = self.beta * vB + (1 - self.beta) * layer.grad_b
        
        layer.W -= self.lr * vW
        layer.b -= self.lr * vB
        
        self.velocities[layer]["vW"] = vW
        self.velocities[layer]["vB"] = vB
        
class RMSProp:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.beta= beta
        self.epsilon = epsilon
        self.cache = {}
        
    def update(self, layer):
        if layer not in self.cache: 
            self.cache[layer] = {
                "sW": np.zeros_like(layer.W),
                "sB": np.zeros_like(layer.b)
            }
        gW = layer.grad_W
        gB= layer.grad_b
        
        self.cache[layer]["sW"] = self.beta * self.cache[layer]["sW"] + (1 - self.beta) * (gW ** 2)
        self.cache[layer]["sB"] = self.beta * self.cache[layer]["sB"] + (1 - self.beta) * (gB ** 2)         
            
        layer.W -= self.lr * gW / (np.sqrt(self.cache[layer]["sW"]) + self.epsilon)
        layer.b -= self.lr * gB / (np.sqrt(self.cache[layer]["sB"]) + self.epsilon)
            
class NAG:
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.velocities = {}
        
    def update(self, layer):
        if layer not in self.velocities:
            self.velocities[layer] = {
                "vW": np.zeros_like(layer.W),
                "vB": np.zeros_like(layer.b)
            }
            
        vW = self.velocities[layer]["vW"]
        vB = self.velocities[layer]["vB"]
            
        vW = self.beta * vW + (1 - self.beta) * layer.grad_W
        vB = self.beta * vB + (1 - self.beta) * layer.grad_b
            
        layer.W -= self.lr * vW
        layer.b -= self.lr * vB
            
        self.velocities[layer]["vW"] = vW
        self.velocities[layer]["vB"] = vB


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.state = {}
        self.t = 0

    def update(self, layer):
        if layer not in self.state:
            self.state[layer] = {
                "mW": np.zeros_like(layer.W), "mB": np.zeros_like(layer.b),
                "vW": np.zeros_like(layer.W), "vB": np.zeros_like(layer.b),
            }
        self.t += 1
        s = self.state[layer]
        s["mW"] = self.beta1 * s["mW"] + (1 - self.beta1) * layer.grad_W
        s["mB"] = self.beta1 * s["mB"] + (1 - self.beta1) * layer.grad_b
        s["vW"] = self.beta2 * s["vW"] + (1 - self.beta2) * (layer.grad_W ** 2)
        s["vB"] = self.beta2 * s["vB"] + (1 - self.beta2) * (layer.grad_b ** 2)
        mW_hat = s["mW"] / (1 - self.beta1 ** self.t)
        mB_hat = s["mB"] / (1 - self.beta1 ** self.t)
        vW_hat = s["vW"] / (1 - self.beta2 ** self.t)
        vB_hat = s["vB"] / (1 - self.beta2 ** self.t)
        layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
        layer.b -= self.lr * mB_hat / (np.sqrt(vB_hat) + self.epsilon)


class NAdam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.state = {}
        self.t = 0

    def update(self, layer):
        if layer not in self.state:
            self.state[layer] = {
                "mW": np.zeros_like(layer.W), "mB": np.zeros_like(layer.b),
                "vW": np.zeros_like(layer.W), "vB": np.zeros_like(layer.b),
            }
        self.t += 1
        s = self.state[layer]
        s["mW"] = self.beta1 * s["mW"] + (1 - self.beta1) * layer.grad_W
        s["mB"] = self.beta1 * s["mB"] + (1 - self.beta1) * layer.grad_b
        s["vW"] = self.beta2 * s["vW"] + (1 - self.beta2) * (layer.grad_W ** 2)
        s["vB"] = self.beta2 * s["vB"] + (1 - self.beta2) * (layer.grad_b ** 2)
        mW_hat = s["mW"] / (1 - self.beta1 ** self.t)
        mB_hat = s["mB"] / (1 - self.beta1 ** self.t)
        vW_hat = s["vW"] / (1 - self.beta2 ** self.t)
        vB_hat = s["vB"] / (1 - self.beta2 ** self.t)
        # Nesterov component
        mW_nesterov = self.beta1 * mW_hat + (1 - self.beta1) * layer.grad_W / (1 - self.beta1 ** self.t)
        mB_nesterov = self.beta1 * mB_hat + (1 - self.beta1) * layer.grad_b / (1 - self.beta1 ** self.t)
        layer.W -= self.lr * mW_nesterov / (np.sqrt(vW_hat) + self.epsilon)
        layer.b -= self.lr * mB_nesterov / (np.sqrt(vB_hat) + self.epsilon)