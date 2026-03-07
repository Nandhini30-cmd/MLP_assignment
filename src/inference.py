"""
Inference Script
Evaluate trained models on test sets
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data 
from ann.objective_functions import CrossEntropy, MSE
import argparse
import numpy as np

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=['mnist', 'fashion_mnist'], help="Dataset to evaluate on")
    parser.add_argument("-b", "--batch_size", type=int, required=True, help="Batch size for inference")
    parser.add_argument("-nhl", "--num_layers", type=int, required=True, help="Number of hidden layers (<=6)")
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", required=True, help="Number of neurons in each hidden layer (<= 128)")
    parser.add_argument("-a", "--activation", type=str, required=True, choices=["relu", "sigmoid", "tanh"], help="Activation function for hidden layers")
    parser.add_argument("-l", "--loss", type=str,  choices=["mse", "cross_entropy"], help="Loss function to use")
    parser.add_argument("-o", "--optimizer", type=str, choices=['SGD', 'momentum', 'rmsprop', 'nag'], help="Optimizer to use")
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate (not needed for inference)")  
    parser.add_argument("-wd", "--weight_decay", type=float, help="Weight decay for L2 regularization")  
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "xavier"], help="Weight initialization method")
    parser.add_argument("-w_p", "--wandb_project", type=str, help="W&B project name for logging ") 
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to saved model weights")
    return parser.parse_args()


def load_model(model_save_path):
    """
    Load trained model from disk.
    """
    weights = np.load(model_save_path, allow_pickle=True).item()
    return weights
    pass


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    logits = model.forward(X_test)
    if len(y_test.shape) > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test.astype(int)
    if model.loss_fn.__class__.__name__ == "CrossEntropy":
        loss = model.loss_fn.forward(logits, y_true)
    else:
        y_true_one_hot = np.zeros_like(logits)
        y_true_one_hot[np.arange(len(y_true)), y_true] = 1
        loss = model.loss_fn.forward(logits, y_true_one_hot)
    y_pred = np.argmax(logits, axis=1)
    if len(y_test.shape) > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test.astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    
    args = parse_arguments()
    model=NeuralNetwork(args)
    if len(args.hidden_size) != args.num_layers:
        raise ValueError("Number of hidden layer sizes must match num_layers")
    # Load dataset
    X_train, y_train, X_test, y_test = load_data(args.dataset)  

    # Load model weights
    weights = load_model(args.model_path)
    model.set_weights(weights)  
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:{results['recall']:.4f}")
    print("Evaluation complete!")
    
    return results


if __name__ == '__main__':
    main()
