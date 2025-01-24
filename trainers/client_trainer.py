import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from collections import OrderedDict
import torch.nn.functional as F

def train_client(model, train_loader, round_num, client_id, config, logger):
    """Train a client model"""
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    
    # Warmup learning rate
    if round_num <= config.WARMUP_ROUNDS:
        actual_lr = config.LEARNING_RATE * (round_num / config.WARMUP_ROUNDS)
    else:
        actual_lr = config.LEARNING_RATE
    
    optimizer = optim.AdamW(model.parameters(), lr=actual_lr, 
                          weight_decay=config.WEIGHT_DECAY, amsgrad=True)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.EPOCHS//2, T_mult=1, eta_min=actual_lr/10
    )
    
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    best_model_state = None
    moving_avg_loss = None
    alpha = 0.9  # For moving average
    
    for epoch in range(config.EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.cuda()
            target = target.squeeze().cuda()
            
            # Mixup augmentation with probability
            if np.random.random() < config.MIXUP_PROB:
                lam = np.random.beta(config.MIXUP_ALPHA, config.MIXUP_ALPHA)
                index = torch.randperm(data.size(0)).cuda()
                mixed_x = lam * data + (1 - lam) * data[index]
                
                optimizer.zero_grad()
                outputs = model(mixed_x)
                loss = lam * criterion(outputs, target) + (1 - lam) * criterion(outputs, target[index])
            else:
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(epoch + batch_idx / len(train_loader))
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # Update moving average loss
        if moving_avg_loss is None:
            moving_avg_loss = epoch_loss
        else:
            moving_avg_loss = alpha * moving_avg_loss + (1 - alpha) * epoch_loss
        
        logger.log_training_loss(round_num, client_id, epoch, epoch_loss)
        logger.logger.debug(f"Round {round_num}, Client {client_id}, Epoch {epoch}: "
                          f"Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%")
        
        # Save best model based on moving average loss
        if moving_avg_loss < best_loss:
            best_loss = moving_avg_loss
            best_model_state = OrderedDict()
            for key, value in model.state_dict().items():
                best_model_state[key] = value.detach().cpu()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= patience:
            logger.logger.debug(f"Early stopping triggered for client {client_id} at epoch {epoch}")
            break
    
    # Return best model state or current state if no best was found
    if best_model_state is None:
        best_model_state = OrderedDict()
        for key, value in model.state_dict().items():
            best_model_state[key] = value.detach().cpu()
    
    return best_model_state
