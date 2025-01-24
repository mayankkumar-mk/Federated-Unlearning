import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import os
import warnings

from config.config import Config
from data.dataset import get_datasets
from data.data_distributor import distribute_data
from models.resnet import FedResNet
from trainers.client_trainer import train_client
from trainers.unlearning_trainer import unlearn_client
from utils.logger import CustomLogger
from utils.metrics import calculate_class_accuracy, federated_averaging
from utils.visualization import (plot_class_accuracies, plot_loss_curve, 
                               plot_accuracy_trend)

# Suppress warnings
warnings.filterwarnings('ignore')

# Enable deterministic behavior
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True

def main():
    # Initialize configuration and logger
    config = Config()
    logger = CustomLogger()
    set_seed()

    logger.logger.info("Starting federated learning with unlearning capabilities...")

    try:
        # Load datasets
        trainset, testset = get_datasets(config)
        test_loader = DataLoader(testset, batch_size=config.BATCH_SIZE, 
                               shuffle=False, num_workers=config.NUM_WORKERS, 
                               pin_memory=True)

        # Initialize global model
        logger.logger.info("Initializing model...")
        global_model = FedResNet(config).cuda()
        best_acc = 0.0

        # Train or load pre-trained model
        if os.path.exists(config.SAVE_MODEL_PATH):
            logger.logger.info("Loading pre-trained model...")
            state_dict = torch.load(config.SAVE_MODEL_PATH)
            global_model.load_state_dict(state_dict)
        else:
            logger.logger.info("Starting federated training...")
            # Distribute data among clients
            client_data_indices = distribute_data(trainset, config)

            # Federated Learning Training
            for round_num in range(config.ROUNDS):
                logger.logger.info(f"\nFederated Round {round_num + 1}/{config.ROUNDS}")
                
                # Select clients for this round
                selected_clients = np.random.choice(
                    range(config.NUM_CLIENTS), 
                    config.CLIENTS_PER_ROUND, 
                    replace=False
                )
                
                local_state_dicts = []
                
                # Train selected clients
                for client_idx in tqdm(selected_clients, desc='Training clients'):
                    # Prepare client dataset
                    client_dataset = Subset(trainset, client_data_indices[client_idx])
                    client_loader = DataLoader(
                        client_dataset, 
                        batch_size=config.BATCH_SIZE,
                        shuffle=True, 
                        num_workers=config.NUM_WORKERS,
                        pin_memory=True
                    )

                    # Initialize and train client model
                    client_model = FedResNet(config).cuda()
                    client_model.load_state_dict(global_model.state_dict())
                    
                    local_state_dict = train_client(
                        client_model, 
                        client_loader, 
                        round_num + 1,
                        client_idx, 
                        config, 
                        logger
                    )
                    local_state_dicts.append(local_state_dict)

                # Average models
                global_state_dict = federated_averaging(local_state_dicts)
                global_model.load_state_dict(global_state_dict)

                # Evaluate global model
                current_acc = calculate_class_accuracy(global_model, test_loader).mean()
                logger.log_round_accuracy(round_num + 1, current_acc)

                # Save best model
                if current_acc > best_acc:
                    best_acc = current_acc
                    torch.save(global_state_dict, config.SAVE_MODEL_PATH)
                    logger.logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

        # Calculate pre-unlearning accuracy
        logger.logger.info("\nCalculating pre-unlearning performance...")
        pre_unlearning_acc = calculate_class_accuracy(global_model, test_loader)
        logger.log_class_accuracy("Pre-unlearning", pre_unlearning_acc)

        # Prepare for unlearning
        logger.logger.info("\nPreparing for unlearning phase...")
        client_data_indices = distribute_data(trainset, config)
        local_unlearned_states = []

        # Federated Unlearning
        logger.logger.info("\nStarting federated unlearning...")
        for client_idx in tqdm(range(config.NUM_CLIENTS), desc='Unlearning'):
            client_dataset = Subset(trainset, client_data_indices[client_idx])
            
            # Split client data into retain and forget sets
            retain_indices = []
            forget_indices = []
            
            for idx in client_data_indices[client_idx]:
                if trainset.labels[idx] in config.CLASSES_TO_UNLEARN:
                    forget_indices.append(idx)
                else:
                    retain_indices.append(idx)

            if len(forget_indices) == 0 or len(retain_indices) == 0:
                logger.logger.info(f"Skipping client {client_idx} - No data to unlearn or retain")
                continue

            # Create data loaders for retain and forget sets
            retain_dataset = Subset(trainset, retain_indices)
            forget_dataset = Subset(trainset, forget_indices)

            retain_loader = DataLoader(
                retain_dataset,
                batch_size=min(config.BATCH_SIZE, len(retain_dataset)),
                shuffle=True,
                num_workers=config.NUM_WORKERS,
                pin_memory=True
            )
            forget_loader = DataLoader(
                forget_dataset,
                batch_size=min(config.BATCH_SIZE, len(forget_dataset)),
                shuffle=True,
                num_workers=config.NUM_WORKERS,
                pin_memory=True
            )

            # Initialize client model
            client_model = FedResNet(config).cuda()
            client_model.load_state_dict(global_model.state_dict())

            # Perform unlearning
            unlearned_state = unlearn_client(
                client_model,
                retain_loader,
                forget_loader,
                client_idx,
                config,
                logger
            )
            local_unlearned_states.append(unlearned_state)

        # Average and save unlearned models
        if local_unlearned_states:
            logger.logger.info("\nAggregating unlearned models...")
            final_state_dict = federated_averaging(local_unlearned_states)
            global_model.load_state_dict(final_state_dict)

            torch.save(final_state_dict, os.path.join(logger.log_dir, config.SAVE_UNLEARNED_MODEL_PATH))
            logger.logger.info(f"Unlearned model saved")

        # Evaluate unlearning effectiveness
        logger.logger.info("\nEvaluating unlearning effectiveness...")
        post_unlearning_acc = calculate_class_accuracy(global_model, test_loader)
        logger.log_class_accuracy("Post-unlearning", post_unlearning_acc)

        # Generate visualizations
        plot_class_accuracies(pre_unlearning_acc, post_unlearning_acc, 
                            config.CLASSES_TO_UNLEARN, logger)
        plot_loss_curve(logger.metrics, logger)
        plot_accuracy_trend(logger.metrics, logger)

        # Print detailed results
        logger.logger.info("\nDetailed Results:")
        logger.logger.info("=" * 50)

        # Calculate metrics
        retain_classes = np.ones(config.NUM_CLASSES, dtype=bool)
        retain_classes[config.CLASSES_TO_UNLEARN] = False

        retain_acc_before = pre_unlearning_acc[retain_classes].mean()
        retain_acc_after = post_unlearning_acc[retain_classes].mean()

        # Log overall results
        logger.logger.info("\nOverall Accuracy (Including forget classes):")
        logger.logger.info(f"Before unlearning: {pre_unlearning_acc.mean():.4f}")
        logger.logger.info(f"After unlearning: {post_unlearning_acc.mean():.4f}")
        logger.logger.info(f"Change: {post_unlearning_acc.mean() - pre_unlearning_acc.mean():.4f}")

        logger.logger.info("\nRetain Set Accuracy:")
        logger.logger.info(f"Before unlearning: {retain_acc_before:.4f}")
        logger.logger.info(f"After unlearning: {retain_acc_after:.4f}")
        logger.logger.info(f"Change: {retain_acc_after - retain_acc_before:.4f}")

        # Log per-class results for forgotten classes
        logger.logger.info("\nUnlearned Classes Performance:")
        for class_idx in config.CLASSES_TO_UNLEARN:
            logger.logger.info(f"\nClass {class_idx}:")
            logger.logger.info(f"Before unlearning: {pre_unlearning_acc[class_idx]:.4f}")
            logger.logger.info(f"After unlearning: {post_unlearning_acc[class_idx]:.4f}")
            logger.logger.info(f"Change: {post_unlearning_acc[class_idx] - pre_unlearning_acc[class_idx]:.4f}")

        # Save final metrics
        logger.save_metrics()

    except Exception as e:
        logger.log_error(str(e))
        raise

if __name__ == "__main__":
    main()
