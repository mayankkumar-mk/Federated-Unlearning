import torchvision.transforms as transforms
from medmnist import PathMNIST

def get_datasets(config):
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    trainset = PathMNIST(split='train', transform=train_transform, download=True)
    testset = PathMNIST(split='test', transform=test_transform, download=True)
    
    return trainset, testset
