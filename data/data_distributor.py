import numpy as np
import matplotlib.pyplot as plt

def distribute_data(dataset, config):
    labels = dataset.labels
    if len(labels.shape) > 1:
        labels = labels.squeeze()
    
    client_data_idx = [[] for _ in range(config.NUM_CLIENTS)]
    label_indices = [np.where(labels == y)[0] for y in range(config.NUM_CLASSES)]
    
    # Ensure minimum samples per class per client
    min_samples_per_class = 3
    for label_idx in range(config.NUM_CLASSES):
        indices = label_indices[label_idx]
        np.random.shuffle(indices)
        samples_per_client = max(min_samples_per_class, len(indices) // config.NUM_CLIENTS)
        
        for client_idx in range(config.NUM_CLIENTS):
            if len(indices) >= samples_per_client:
                selected_indices = indices[:samples_per_client]
                client_data_idx[client_idx].extend(selected_indices)
                indices = indices[samples_per_client:]
    
    # Distribute remaining data using Dirichlet
    remaining_indices = [idx for class_indices in label_indices for idx in class_indices 
                        if not any(idx in client_indices for client_indices in client_data_idx)]
    
    if remaining_indices:
        remaining_labels = labels[remaining_indices]
        client_distributions = np.random.dirichlet(config.DIRICHLET_ALPHA * np.ones(config.NUM_CLIENTS), 
                                                 config.NUM_CLASSES)
        
        for label_idx in range(config.NUM_CLASSES):
            label_mask = (remaining_labels == label_idx)
            class_indices = np.array(remaining_indices)[label_mask]
            
            if len(class_indices) > 0:
                class_distribution = client_distributions[label_idx]
                num_samples_per_client = (class_distribution * len(class_indices)).astype(int)
                num_samples_per_client[-1] = len(class_indices) - num_samples_per_client[:-1].sum()
                
                start_idx = 0
                for client_idx, num_samples in enumerate(num_samples_per_client):
                    if num_samples > 0:
                        end_idx = start_idx + num_samples
                        client_data_idx[client_idx].extend(class_indices[start_idx:end_idx])
                        start_idx = end_idx
    
    plot_distribution(dataset, client_data_idx, config)
    return client_data_idx

def plot_distribution(dataset, client_data_idx, config):
    labels = dataset.labels.squeeze()
    distribution = np.zeros((config.NUM_CLASSES, config.NUM_CLIENTS))
    
    for client_idx, indices in enumerate(client_data_idx):
        client_labels = labels[indices]
        unique_labels, counts = np.unique(client_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            distribution[int(label), client_idx] = count
    
    plt.figure(figsize=(15, 8))
    plt.imshow(distribution, aspect='auto', cmap='YlOrRd')
    
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLIENTS):
            if distribution[i, j] > 0:
                plt.text(j, i, int(distribution[i, j]), 
                        ha='center', va='center',
                        color='black' if distribution[i, j] < distribution.max()/2 else 'white')
    
    plt.colorbar(label='Number of Samples')
    plt.xlabel('Client Number')
    plt.ylabel('Class')
    plt.title('Data Distribution Across Clients and Classes')
    plt.xticks(range(config.NUM_CLIENTS))
    plt.yticks(range(config.NUM_CLASSES))
    plt.tight_layout()
    plt.savefig('data_distribution.png')
    plt.close()
