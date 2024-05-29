import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

def prepare_data(batch_size, image_size_x, image_size_y):
    train_dir = os.path.join('Data', 'train')
    valid_dir = os.path.join('Data', 'valid')
    train_transformer = transforms.Compose([
        transforms.Resize((image_size_x, image_size_y)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    test_transformer = transforms.Compose([
        transforms.Resize((image_size_x, image_size_y)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    trainloader = DataLoader(torchvision.datasets.ImageFolder(train_dir,transform=train_transformer), 
                             batch_size=batch_size, shuffle=True, num_workers=7, persistent_workers=True)
    testloader = DataLoader(torchvision.datasets.ImageFolder(valid_dir,transform=test_transformer), 
                            batch_size=batch_size, shuffle=False, num_workers=7, persistent_workers=True)

    return trainloader, testloader
