"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
import json
import os

import numpy as np
from ann.neural_network import NeuralNetwork
from ann.neural_layer import NeuralLayer
from ann.optimizers import SGD, Momentum, RMSProp, NAG
from inference import evaluate_model
from utils.data_loader import load_data
from ann.objective_functions import get_loss_function
import argparse
import wandb


def save_model(model, save_path):
    """
    Save model weights to disk.
    """
    if save_path is None:
        return 
    weights = model.get_weights()
    np.save(save_path, weights)
    print(f"Model saved at {save_path}")
    
def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    # Dataset
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=['mnist', 'fashion_mnist'], help="Dataset to use")
    
    # Epochs and Batch Size
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, required=True, help="Mini-batch size")
    
    # Loss and Optimizer
    parser.add_argument("-l", "--loss", type=str, required=True, choices= ["mse", "cross_entropy"], help="Loss to use")
    parser.add_argument("-o", "--optimizer", type=str, required=True, choices=['SGD', 'sgd', 'momentum', 'rmsprop', 'nag'], help="Optimizer to use")
    
    # Learning Rate and Weight decay
    parser.add_argument('-lr',"--learning_rate", type = float, required=True, help = "Initial Learning rate to use")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0, help="Weight decay value for L2 regularization")        
    
    # Number of Hidden Layers and Neurons
    parser.add_argument("-nhl", "--num_layers", type=int, required=True, help="Number of hidden layers (<=6)")
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", required=True, help="Number of neurons in each hidden layer (<= 128)")
    
    # Activation Function and Weight Initialization
    parser.add_argument("-a", "--activation", type=str, required=True, choices=["relu", "sigmoid", "tanh"], help="Activation function for hidden layers")
    parser.add_argument("-w_i", "--weight_init", type=str, required=True, choices=["random", "xavier"], help="Weight initialization method")
    
    # Optional: W&B logging and Model Saving
    parser.add_argument("-w_p", "--wandb_project", type=str, help="W&B project name for logging")
    parser.add_argument("-m", "--model_save_path", type=str, help="Path to save the trained model")
    
    return parser.parse_args()

def load_model(model_path):
        data = np.load(model_path, allow_pickle=True).item()
        return data
    
def main():
    """
    Main training function.
    """
    args = parse_arguments()
    if args.wandb_project:
        wandb.init(project=args.wandb_project, config = vars(args))
        print(f"W&B logging enabled for project: {args.wandb_project}")
    else:
        print("W&B logging not enabled. To enable, provide --wandb_project argument.")  
    # Basic argument validation
    if len(args.hidden_size) != args.num_layers:
        raise ValueError(
            "Number of hidden sizes must match num_layers"
        )

    if args.num_layers > 6:
        raise ValueError("Number of hidden layers must be <= 6")

    if any(h > 128 for h in args.hidden_size):
        raise ValueError("Hidden layer size must be <= 128")

    print("Configuration:")
    print(args)

    # Load dataset
    X_train, y_train, X_test, y_test = load_data(args.dataset)
    
    if y_train.ndim ==2:
        y_train=np.argmax(y_train, axis=1)
        
    if y_test.ndim ==2:
        y_test = np.argmax(y_test, axis =1)
        
    y_train= y_train.astype(int)
    y_test = y_test.astype(int)
    # Build model
    model = NeuralNetwork(args)
    
    if args.model_save_path and os.path.exists(args.model_save_path):
        print(f"Loading existing model weighrs..")
        weights = load_model(args.model_save_path)
        model.set_weights(weights)

    # Train
    best_f1 =0
    best_weights = None
    best_config = vars(args)
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        model.train(X_train, y_train, 1, args.batch_size)
        results = evaluate_model(model, X_test, y_test)
        f1 = results["f1"]
        print(f"Test F1:{f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_weights = model.get_weights()
            
    np.save("best_model.npy", best_weights)
    with open("best_config.json", "w") as f:
        json.dump(best_config, f, indent=4)
    print("Best model.npy and best_config.json saved in src folder.")
    
    print("Training complete!")

if __name__ == '__main__':
    main()
