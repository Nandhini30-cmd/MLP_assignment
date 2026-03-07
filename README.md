# DA6401 Assignment 1: Multi-Layer Perceptron for Image Classification

A neural network built **from scratch using NumPy** for image classification on MNIST and Fashion-MNIST datasets, with full [Weights & Biases](https://wandb.ai/) experiment tracking.

---

## Project Structure

```
├── src/
│   ├── ann/                      # Core neural network library
│   │   ├── neural_network.py     # NeuralNetwork class (forward, backprop, training loop)
│   │   ├── neural_layer.py       # Dense layer (weights, biases, gradients)
│   │   ├── activations.py        # ReLU, Sigmoid, Tanh & their derivatives
│   │   ├── optimizers.py         # SGD, Momentum, NAG, RMSProp
│   │   └── objective_functions.py# Cross-Entropy & MSE loss functions
│   ├── utils/
│   │   └── data_loader.py        # Load MNIST / Fashion-MNIST via Keras
│   ├── train.py                  # CLI training entry point
│   ├── inference.py              # CLI inference & evaluation
│   ├── best_config.json          # Best hyperparameter configuration
│   └── best_model.npy            # Best saved model weights
├── notebooks/
│   ├── assignment_report.ipynb   # Full assignment report & experiments notebook
│   └── wandb_demo.ipynb          # W&B demo notebook
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- A [Weights & Biases](https://wandb.ai/) account (for experiment logging)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd da6401_assignment_1-main

# Install dependencies
pip install -r requirements.txt

# Log in to Weights & Biases
wandb login
```

---

## Training

Train a model from the command line:

```bash
cd src

python train.py \
  -d mnist \
  -e 30 \
  -b 32 \
  -l cross_entropy \
  -o rmsprop \
  -lr 0.001 \
  -wd 0.0005 \
  -nhl 3 \
  -sz 128 64 32 \
  -a relu \
  -w_i xavier \
  -w_p mnist_sweep \
  -m best_model.npy
```




### Training Arguments

| Flag | Argument | Description | Choices |
|-----|------|-------------|---------|
| -d | --dataset | Dataset | mnist, fashion_mnist |
| -e | --epochs | Number of epochs | int |
| -b | --batch_size | Mini-batch size | int |
| -l | --loss | Loss function | cross_entropy, mse |
| -o | --optimizer | Optimizer | sgd, momentum, nag, rmsprop |
| -lr | --learning_rate | Learning rate | float |
| -wd | --weight_decay | L2 regularization | float |
| -nhl | --num_layers | Hidden layers | int |
| -sz | --hidden_size | Neurons per layer | list[int] |
| -a | --activation | Activation function | relu, sigmoid, tanh |
| -w_i | --weight_init | Weight initialization | random, xavier |
| -w_p | --wandb_project | W&B project name | string |
| -m | --model_save_path | Model save path | string |

## Inference

Evaluate a saved model on the test set:

```bash
cd src

python src/inference.py \
-d mnist \
-b 32 \
-nhl 3 \
-sz 128 64 32 \
-a relu \
-m src/best_model.npy
```

Output includes  accuracy, F1 score, precision and recall.

---

## Experiments (W&B Report)

All experiments are documented in the Jupyter notebook:

```
notebooks/assignment_report.ipynb
```

Open it in Jupyter Notebook or JupyterLab to view results and re-run experiments:

```bash
jupyter notebook notebooks/assignment_report.ipynb


### Experiment Sections
 #  Experiment  Description 

2.1 Data Exploration  Class distribution analysis 
2.2 Hyperparameter Sweep 100+ run sweep with W&B 
2.3 Optimizer Showdown SGD vs Momentum vs RMSProp vs Nag
2.4 Vanishing Gradients Sigmoid vs ReLU gradient analysis 
2.5 Dead Neurons  ReLU (high LR) vs Tanh investigation
2.6 Loss Comparison MSE vs Cross-Entropy 
2.7 Performance Analysis  Train vs Test accuracy overlay  
2.8 Error Analysis  Confusion matrix & visualizations 
2.9 Weight Initialization Zeros vs Xavier (50 iterations) 
2.10Transfer Learning Fashion-MNIST with 3 configurations 


## Best Configuration

The best hyperparameters found via sweep:

## Best Configuration

```json
{
    "dataset": "mnist",
    "epochs": 30,
    "batch_size": 32,
    "loss": "cross_entropy",
    "optimizer": "rmsprop",
    "learning_rate": 0.001,
    "weight_decay": 0.0005,
    "num_layers": 3,
    "hidden_size": [128, 64, 32],
    "activation": "relu",
    "weight_init": "xavier"
}


## Dependencies

NumPy — Core computations
Matplotlib — Plotting & visualizations
- Keras — Dataset loading (MNIST / Fashion-MNIST)
- Weights & Biases — Experiment tracking & sweeps
- scikit-learn — Metrics (accuracy, F1, precision, recall)

---

## Notes

1. The neural network (forward pass, backpropagation, weight updates) is implemented entirely from scratch using NumPy — no deep learning frameworks are used for the model itself.
2. Keras is used only for loading the MNIST and Fashion-MNIST datasets.
3. All experiments log to W&B for reproducibility and comparison.



---






## Author

Nandhini M (ns26z008)
Applied Mechanics & Biomedical Engineering  
