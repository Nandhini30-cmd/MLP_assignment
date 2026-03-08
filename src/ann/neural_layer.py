"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np 

class NeuralLayer:
    """
    Implements a fully connected (dense) neural layer.
    """
    
    def __init__ (self, input_size, output_size, weight_init="random"):
        
        """
            Initialize weights and bias.

            Args:
                input_size: Number of input features
                output_size: Number of neurons in this layer
                weight_init: 'random' or 'xavier'
            """
    
        if weight_init  == "xavier":
            self.W = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        else:
            self.W = np.random.randn(input_size, output_size) * 0.01
                
        self.b = np.zeros((1,output_size))
        
    def forward(self, X):
        """
        Forward pass.
        
        Args:
            X:Input data of shape (batch_size, input_size)
            
        Returns:
            Z: Linear output
        """
        self.X = X # store input for backward
        self.Z = np.dot(X, self.W) + self.b 
        return self.Z
    
    def backward(self, dZ):
        """
        Backward pass.
        Args:
            dZ: Gradient from next layer (batch_size, output_size)
        Returns:
            dX: Gradient w.r.t input
        """

        m = self.X.shape[0]  # batch size

        # Gradients
        self.grad_W = np.dot(self.X.T, dZ)/m
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / m

        # Gradient w.r.t input (to pass to previous layer)
        dX = np.dot(dZ, self.W.T)

        return dX
        
if __name__ == "__main__":
    layer = NeuralLayer(5, 3)
    X = np.random.randn(4, 5)
    output = layer.forward(X)
    print("Output shape:", output.shape)
    
    


    
    
        
    
    