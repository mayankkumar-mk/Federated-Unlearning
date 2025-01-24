import logging
import json
from pathlib import Path
from datetime import datetime

class CustomLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_loggers(timestamp)
        
        self.metrics = {
            'train_losses': [],
            'round_accuracies': [],
            'class_accuracies': [],
            'unlearning_metrics': {
                'pre_unlearning': None,
                'post_unlearning': None
            }
        }
    
    def setup_loggers(self, timestamp):
        self.logger = logging.getLogger('FederatedLearning')
        self.logger.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        file_handler = logging.FileHandler(self.log_dir / f'federated_learning_{timestamp}.log')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        self.metrics_file = self.log_dir / f'metrics_{timestamp}.json'
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def log_training_loss(self, round_num, client_id, epoch, loss):
        self.logger.info(f"Round {round_num}, Client {client_id}, Epoch {epoch}: Loss = {loss:.4f}")
        self.metrics['train_losses'].append({
            'round': round_num,
            'client_id': client_id,
            'epoch': epoch,
            'loss': loss
        })
    
    def log_round_accuracy(self, round_num, accuracy):
        self.logger.info(f"Round {round_num} Average Accuracy: {accuracy:.4f}")
        self.metrics['round_accuracies'].append({
            'round': round_num,
            'accuracy': accuracy
        })
    
    def log_class_accuracy(self, phase, accuracies):
        self.logger.info(f"\n{phase} Class Accuracies:")
        for class_idx, acc in enumerate(accuracies):
            self.logger.info(f"Class {class_idx}: {acc:.4f}")
        
        if phase == "Pre-unlearning":
            self.metrics['unlearning_metrics']['pre_unlearning'] = accuracies
        elif phase == "Post-unlearning":
            self.metrics['unlearning_metrics']['post_unlearning'] = accuracies
    
    def log_unlearning_progress(self, client_id, epoch, loss):
        self.logger.info(f"Client {client_id} Unlearning Epoch {epoch}: Loss = {loss:.4f}")
    
    def save_metrics(self):
        metrics_to_save = {
            'train_losses': self.metrics['train_losses'],
            'round_accuracies': self.metrics['round_accuracies'],
            'unlearning_metrics': {
                'pre_unlearning': self.metrics['unlearning_metrics']['pre_unlearning'].tolist() 
                    if self.metrics['unlearning_metrics']['pre_unlearning'] is not None else None,
                'post_unlearning': self.metrics['unlearning_metrics']['post_unlearning'].tolist()
                    if self.metrics['unlearning_metrics']['post_unlearning'] is not None else None
            }
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        self.logger.info(f"Metrics saved to {self.metrics_file}")
    
    def log_error(self, error_msg):
        self.logger.error(f"Error occurred: {error_msg}")
