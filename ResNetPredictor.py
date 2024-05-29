import torch
import torchvision.transforms as transforms
from PIL import Image
from model import Net
from customDataset import CustomDataset
import os

train_data_dir = os.path.join('Data','train')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset_object = CustomDataset(root_dir=train_data_dir, transform=transform)

class ResNetPredictor:
    def __init__(self, model_path, lr=0.001):
        self.model = Net(lr)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.class_labels = train_dataset_object.get_label()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # Find the correct key for the model state dictionary
        model_state_dict_key = next(iter(checkpoint))
        self.model.load_state_dict(checkpoint[model_state_dict_key])

    def predict(self, image_path):
        image = Image.open(image_path)
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_index = predicted.item()

            # Debugging: Print the predicted class index
            print("Predicted Index:", predicted_index)

            try:
                predicted_label = self.class_labels[predicted_index]
            except KeyError:
                print("Error: Predicted index not found in class labels.")

        return predicted_label
