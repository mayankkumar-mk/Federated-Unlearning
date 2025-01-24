import torch
from collections import OrderedDict
import numpy as np

def calculate_class_accuracy(model, loader):
    """Calculate accuracy for each class"""
    class_correct = torch.zeros(model.module.config.NUM_CLASSES if hasattr(model, 'module') 
                              else model.config.NUM_CLASSES).cuda()
    class_total = torch.zeros_like(class_correct)
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.squeeze().cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            for i in range(len(targets)):
                label = targets[i]
                class_correct[label] += (predicted[i] == label).float()
                class_total[label] += 1
    
    class_accuracy = (class_correct / class_total).cpu().numpy()
    return class_accuracy

def federated_averaging(state_dicts):
    """Average model parameters"""
    averaged_dict = OrderedDict()
    
    for key in state_dicts[0].keys():
        stacked = torch.stack([state_dict[key].float() for state_dict in state_dicts])
        averaged_dict[key] = stacked.mean(dim=0)
        
        # Convert back to original dtype if needed
        if state_dicts[0][key].dtype != torch.float32:
            averaged_dict[key] = averaged_dict[key].to(state_dicts[0][key].dtype)
    
    return averaged_dict
