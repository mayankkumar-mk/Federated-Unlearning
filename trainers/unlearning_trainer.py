import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import OrderedDict

def calculate_shapley_values(model, data_loader, config):
    model_arr, model_slist = get_net_arr(model)
    num_neurons = len(model_arr)
    shapley_values = torch.zeros(num_neurons).numpy()
    
    for x, y in data_loader:
        x, y = x.cuda(), y.squeeze().cuda()
        for _ in range(config.SHAPLEY_REPEATS):
            perm = random.sample(range(num_neurons), int(num_neurons*0.25))
            zeroed_neurons = torch.ones(num_neurons)
            zeroed_neurons[list(perm)] = 0
            zeroed_model = np.multiply(model_arr, zeroed_neurons.numpy())
            zeroed_model = get_arr_net(model, zeroed_model, model_slist)
            
            zeroed_output = zeroed_model(x)
            loss = F.cross_entropy(zeroed_output, y)
            loss.backward()
            
            prev_index = 0
            index = 0
            for param in zeroed_model.parameters():
                prev_index = index
                index = index + len(param.flatten())
                if param.grad is not None:
                    grad_np = param.grad.cpu().detach().numpy().flatten()
                    shapley_values[prev_index:index] += np.abs(grad_np * model_arr[prev_index:index])
    
    return shapley_values

def get_net_arr(model):
    param_list = [param.data.cpu().numpy() for param in model.parameters()]
    arr = np.array([])
    slist = []
    
    for index, item in enumerate(param_list):
        slist.append(item.shape)
        item = item.reshape((-1))
        arr = np.concatenate((arr, item)) if arr.size else item
    
    return arr, slist

def get_arr_net(model, arr, slist):
    model_copy = type(model)(model.config).cuda()
    start_index = 0
    state_dict = OrderedDict()
    
    for (name, _), shape in zip(model.named_parameters(), slist):
        size = np.prod(shape)
        param = torch.from_numpy(arr[start_index:start_index + size].reshape(shape)).cuda()
        state_dict[name] = param
        start_index += size
    
    model_copy.load_state_dict(state_dict)
    return model_copy

def unlearn_client(model, retain_loader, forget_loader, client_id, config, logger):
    shapley_values = calculate_shapley_values(model, forget_loader, config)
    
    # Get critical parameters (top 1%)
    num_params = len(shapley_values)
    num_to_zero = int(num_params * config.PRUNE_RATIO)
    critical_indices = np.argpartition(shapley_values, -num_to_zero)[-num_to_zero:]
    
    # Zero out parameters
    model_arr, model_slist = get_net_arr(model)
    model_arr[critical_indices] = 0
    updated_model = get_arr_net(model, model_arr, model_slist)
    
    # Fine-tune on retain set
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(updated_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    for epoch in range(config.UNLEARNING_EPOCHS):
        updated_model.train()
        running_loss = 0.0
        for inputs, targets in retain_loader:
            inputs, targets = inputs.cuda(), targets.squeeze().cuda()
            optimizer.zero_grad()
            outputs = updated_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(retain_loader)
        logger.log_unlearning_progress(client_id, epoch, epoch_loss)
    
    return {k: v.cpu() for k, v in updated_model.state_dict().items()}
