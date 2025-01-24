# Federated Unlearning for Medical Image Classification

This repository contains the implementation of a federated unlearning framework for medical image classification tasks. The framework allows for selective forgetting of specific classes while maintaining performance on retained classes.

## Features
- Federated Learning with ResNet18 backbone
- Non-IID data distribution using Dirichlet sampling
- Selective class unlearning using Shapley values
- Comprehensive logging and visualization
- Support for PathMNIST dataset

## Setup
1. Create a new virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage
Run the main training and unlearning pipeline:
```bash
python main.py
```

## Project Structure
```
federated_unlearning/
├── config/          # Configuration parameters
├── data/            # Dataset and data distribution
├── models/          # Model architectures
├── trainers/        # Training and unlearning logic
├── utils/           # Logging and visualization utilities
└── main.py         # Main execution script
```

## Configuration
Key parameters can be modified in `config/config.py`:
- Number of clients: 10
- Clients per round: 10
- Training rounds: 100
- Dirichlet alpha: 0.1
- Prune ratio: 1%
- Classes to unlearn: [2]