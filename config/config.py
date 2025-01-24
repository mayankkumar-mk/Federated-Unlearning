class Config:
    def __init__(self):
        # Federated Learning Parameters
        self.NUM_CLIENTS = 10
        self.CLIENTS_PER_ROUND = 10
        self.NUM_CLASSES = 9
        self.BATCH_SIZE = 128
        self.NUM_WORKERS = 12
        self.EPOCHS = 5
        self.ROUNDS = 100
        self.LEARNING_RATE = 0.0005
        self.DIRICHLET_ALPHA = 0.1
        self.WARMUP_ROUNDS = 10

        # Unlearning Parameters
        self.PRUNE_RATIO = 0.01  # 1%
        self.UNLEARNING_EPOCHS = 5
        self.CLASSES_TO_UNLEARN = [2]
        self.SHAPLEY_REPEATS = 10

        # Model Paths
        self.SAVE_MODEL_PATH = 'best_federated_model_pathmnist.pth'
        self.SAVE_UNLEARNED_MODEL_PATH = 'unlearned_model_pathmnist.pth'

        # Training Parameters
        self.LABEL_SMOOTHING = 0.1
        self.WEIGHT_DECAY = 0.01
        self.MIXUP_ALPHA = 0.2
        self.MIXUP_PROB = 0.3
