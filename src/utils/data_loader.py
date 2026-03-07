"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from keras.datasets import mnist, fashion_mnist

def one_hot_encode(labels, num_classes=10):
    """
    Convert integer labels to one-hot encoded vectors.
    """
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot

def load_data(dataset_name):
        """
        Load and preprocess the specified dataset.
        
        Args:
            dataset_name: 'mnist' or 'fashion_mnist'
        """
        if dataset_name == "mnist":
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
        elif dataset_name == "fashion_mnist":
            (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        else:
            raise ValueError("Unsupported dataset. Choose 'mnist' or 'fashion_mnist'.")
        
        # Normalize pixel values to [0, 1]
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
        
        # Flatten images to vectors
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        # One-hot encode labels
        y_train = one_hot_encode(y_train)
        y_test = one_hot_encode(y_test)
        
        return X_train, y_train, X_test, y_test     
  