import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class DataLoaderFactory:  
    @staticmethod
    def load_data(dataset_name='MNIST', batch_size=64):
        current_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        if current_dir.endswith('experiments'):
            root = os.path.join(current_dir, '..', 'data')
        else:
            root = os.path.join(current_dir, 'data')
        root = os.path.abspath(root)

        if dataset_name == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  
            ])

            train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        
        elif dataset_name == 'CIFAR10':

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
            ])

            train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
