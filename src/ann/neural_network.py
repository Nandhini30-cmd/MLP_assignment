"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from ann.neural_layer import NeuralLayer
from ann.activations import ReLU, Softmax, Sigmoid, Tanh
from ann.objective_functions import CrossEntropy, MSE
from ann.optimizers import SGD, Momentum, RMSProp, NAG
class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        self.layers = []
        input_size = 784  
        
        for i in range(cli_args.num_layers):
            output_size = cli_args.hidden_size[i] 
            self.layers.append(NeuralLayer(input_size, output_size))
           
            if cli_args.activation.lower() == "relu":
                self.layers.append(ReLU())
            elif cli_args.activation.lower() == "sigmoid":
                self.layers.append(Sigmoid())
            elif cli_args.activation.lower() == "tanh":
                self.layers.append(Tanh())  
    
            input_size = output_size
        
        self.layers.append(NeuralLayer(input_size, 10))  
       
        # Loss
        if cli_args.loss == "mse":
            self.loss_fn = MSE()
        else:
            self.loss_fn = CrossEntropy()

        # Optimizer
        lr = getattr(cli_args, "learning_rate", 0.001)
        opt = cli_args.optimizer.lower()
        if opt == "sgd":
            self.optimizer = SGD(lr)
        elif opt == "momentum":
            self.optimizer = Momentum(lr)
        elif opt == "nag":
            self.optimizer = NAG(lr)
        elif opt == "rmsprop":
            self.optimizer = RMSProp(lr)
        
        
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """
        # Compute loss gradient
        dA = self.loss_fn.backward()
        
        # Backpropagate through layers in reverse order
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
            
        
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        for layer in self.layers:
            if hasattr(layer, "W"): 
                self.optimizer.update(layer)    
    
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            for i in range(0, num_samples, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                # Forward pass
                logits = self.forward(X_batch)
                
                # Compute loss
                if len(y_batch.shape) > 1:
                    y_batch=np.argmax(y_batch,axis=1)
                loss = self.loss_fn.forward(logits, y_batch)
                
                dA = self.loss_fn.backward()
                
                # Backward pass
                for layer in reversed(self.layers):
                    dA = layer.backward(dA)
                
                # Update weights
                self.update_weights()
            
            logits_full = self.forward(X_train)
            if len(y_train.shape) > 1:
                y_train_labels =np.argmax(y_train, axis=1)
            else:
                y_train_labels = y_train.astype(int)
                
            epoch_loss = self.loss_fn.forward(logits_full, y_train_labels)
            epoch_accuracy = np.mean(np.argmax(logits_full, axis=1) == y_train_labels) * 100
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    
    def get_weights(self):
        """
        Get model weights for saving.
        """
        weights = {}
        layer_index = 0
        for layer in self.layers:
            if hasattr(layer, "W") and hasattr(layer, "b"):
                weights[f"W{layer_index}"] = layer.W
                weights[f"b{layer_index}"] = layer.b
                layer_index += 1
        return weights
    
    def set_weights(self, weights):
        """
        Set model weights from loaded data.
        """
        layer_index = 0
        for layer in self.layers:
            if hasattr(layer, "W") and hasattr(layer, "b"):
                layer.W = weights[f"W{layer_index}"]
                layer.b = weights[f"b{layer_index}"]
                layer_index += 1
    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        y_pred = self.forward(X)
        predictions = np.argmax(y_pred, axis=1)
        if len(y.shape) > 1:
            true_labels = np.argmax(y, axis=1)
        else:
            true_labels = y.astype(int)
        accuracy = np.mean(predictions == true_labels)
        print(f"Test Accuracy: {accuracy:.4f}")
        loss = self.loss_fn.forward(y_pred, true_labels)
        print(f"Test Loss: {loss:.4f}")