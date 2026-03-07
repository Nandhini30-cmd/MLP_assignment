"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""


import numpy as np

# CROSS ENTROPY LOSS

class CrossEntropy:
    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, logits, y_true):
        """
        logits: (N, C) raw outputs from last layer
        y_true: (N,) integer class labels
        """

        # Numerical stability trick
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)

        exp_scores = np.exp(shifted_logits)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        self.y_true = y_true
        N = logits.shape[0]

        # Compute cross-entropy loss
        correct_log_probs = -np.log(self.probs[range(N), y_true] + 1e-15)
        loss = np.mean(correct_log_probs)

        return loss

    def backward(self):
        """
        Returns gradient w.r.t logits
        """
        N = self.probs.shape[0]

        dlogits = self.probs.copy()
        dlogits[range(N), self.y_true] -= 1
        dlogits /= N

        return dlogits


# MEAN SQUARED ERROR LOSS

class MSE:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        """
        y_pred: (N, C)
        y_true: (N, C) one-hot encoded
        """
        self.y_pred = y_pred
        self.y_true = y_true

        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self):
        """
        Returns gradient w.r.t predictions
        """
        N = self.y_pred.shape[0]
        grad = (2 / N) * (self.y_pred - self.y_true)
        return grad

def get_loss_function(loss_name):
    if loss_name == "cross_entropy":
        return CrossEntropy()
    elif loss_name == "mse":
        return MSE()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")