from PIL import ImageFile
import torch
from train import train_lightning

configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters':
    {
        'model_type': {'value': 'precise'},
        'training_type': {'value': 'lightning'},
        'image_size_x': {'value': 256},
        'image_size_y': {'value': 256},
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [5, 10, 15]},
        'lr': {'max': 0.001, 'min': 0.00001},
        'patience': {'value': 10}
    }
}

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(config):
    print("Training with sweep and configuration file")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using torch device of type {device.type}{": " + torch.cuda.get_device_name(device) if device.type == "cuda" else ""}')

    print("Start Training...")
    # Extract values from config dictionary
    model_type = 'precise'
    lr = 0.00005
    batch_size = 16
    n_epoch = 15
    image_size_x = 256
    image_size_y = 256
    patience = 10

    train_lightning(image_size_x, image_size_y,
                    batch_size, lr, n_epoch, patience)


# Run main function with the configuration
main(configuration)
