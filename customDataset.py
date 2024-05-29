import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._collect_image_paths()
        self.labels, self.label_to_idx = self._find_labels()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Extract label from the directory name
        label = os.path.basename(os.path.dirname(img_path))
        label_idx = self.label_to_idx[label]

        return image, label_idx

    def to_dataframe(self):
        data = {
            'label': [os.path.basename(os.path.dirname(path)) for path in self.image_paths],
            'label_idx': [self.label_to_idx[os.path.basename(os.path.dirname(path))] for path in self.image_paths],
            'image_path': self.image_paths
        }
        return pd.DataFrame(data)

    def index_to_label(self, index):
        return self.labels[index]

    def _collect_image_paths(self):
        image_paths = []
        for subdir, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(subdir, filename))
        return image_paths

    def _find_labels(self):
        labels = sorted(os.listdir(self.root_dir))
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        return labels, label_to_idx

    def get_label(self):
        input_dict = self._find_labels()[1]
        return {value: key for key, value in input_dict.items()}