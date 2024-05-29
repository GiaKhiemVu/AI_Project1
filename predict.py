from customDataset import CustomDataset  # Assuming CustomDataset class is defined in customDataset.py
from ResNetPredictor import ResNetPredictor
from torchvision import transforms
import os

train_data_dir = os.path.join('Data','train')
test_data_dir = os.path.join('Data','valid')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset_object = CustomDataset(root_dir=train_data_dir, transform=transform)
test_dataset_object = CustomDataset(root_dir=test_data_dir, transform=transform)

train_dataset = train_dataset_object.to_dataframe()
test_dataset = test_dataset_object.to_dataframe()

index_picture = 9
print(f'Picture index: {index_picture} at the path {test_dataset.iloc[index_picture]['image_path']}')
print(f'Actual idx: {test_dataset.iloc[index_picture]['label_idx']}\nActual Label: {test_dataset.iloc[index_picture]['label']}')
image_path = test_dataset.iloc[index_picture]['image_path']

model_path = os.path.join('ResNet','model_trained','trained_model.pt')
ckpt_path = os.path.join('ResNet','model_trained', 'epoch=14-step=24240.ckpt')

resnet_predictor = ResNetPredictor(model_path)
predicted_label = resnet_predictor.predict(image_path)
print("Predicted Label:", predicted_label)
