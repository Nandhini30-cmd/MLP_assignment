import numpy as np
import json
import os
import argparse
from sklearn.metrics import f1_score
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():
    parser = argparse.ArgumentParser(description='Test a trained neural network')
    parser.add_argument("-d", "--dataset", type=str, choices=['mnist', 'fashion_mnist'], help="Dataset to evaluate on")
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size for inference")
    parser.add_argument("-nhl", "--num_layers", type=int, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", help="Number of neurons in each hidden layer")
    parser.add_argument("-a", "--activation", type=str, choices=["relu", "sigmoid", "tanh"], help="Activation function")
    parser.add_argument("-l", "--loss", type=str, choices=["mse", "cross_entropy"], help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str, choices=['sgd', 'momentum', 'rmsprop', 'nag'], help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate")
    parser.add_argument("-wd", "--weight_decay", type=float, help="Weight decay")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "xavier"], help="Weight initialization")
    parser.add_argument("-w_p", "--wandb_project", type=str, help="W&B project name")
    parser.add_argument("-m", "--model_path", type=str, help="Path to saved model weights")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # Load defaults from best_config.json if it exists
    config = {}
    for config_path in ["best_config.json", "src/best_config.json"]:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            break

    # Use CLI args if provided, otherwise fall back to config, then to hardcoded defaults
    dataset = args.dataset or config.get("dataset", "mnist")
    num_layers = args.num_layers or config.get("num_layers", 3)
    hidden_size = args.hidden_size or config.get("hidden_size", [128, 64, 32])
    activation = args.activation or config.get("activation", "relu")
    loss = args.loss or config.get("loss", "cross_entropy")
    optimizer = args.optimizer or config.get("optimizer", "sgd")
    learning_rate = args.learning_rate or config.get("learning_rate", 0.001)
    weight_decay = args.weight_decay if args.weight_decay is not None else config.get("weight_decay", 0.0)
    weight_init = args.weight_init or config.get("weight_init", "xavier")
    batch_size = args.batch_size or config.get("batch_size", 32)

    # Determine model path
    model_path = args.model_path
    if model_path is None:
        for mp in ["best_model.npy", "src/best_model.npy"]:
            if os.path.exists(mp):
                model_path = mp
                break
    if model_path is None:
        model_path = "best_model.npy"

    # Build config namespace for NeuralNetwork
    model_config = argparse.Namespace(
        dataset=dataset,
        num_layers=num_layers,
        hidden_size=hidden_size,
        activation=activation,
        loss=loss,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        weight_init=weight_init,
        batch_size=batch_size,
    )

    # Build model with same architecture
    model = NeuralNetwork(model_config)

    # Load saved weights
    weights = np.load(model_path, allow_pickle=True).item()
    model.set_weights(weights)

    # Load actual test data
    X_train, y_train, X_test, y_test = load_data(dataset)

    if y_test.ndim == 2:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test.astype(int)

    # Forward pass
    y_pred = model.forward(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    f1 = f1_score(y_true, y_pred_labels, average='macro')
    print(f"F1 Score: {f1}")