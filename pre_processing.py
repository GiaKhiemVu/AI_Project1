import os
from PIL import Image

def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError) as e:
        return False


def clean_truncated_images(data_dir):
    print(data_dir)
    corrupted_files = []
    # Check both 'train' and 'valid' folders
    #Data
    for subdir in ['train', 'valid']:
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.exists(subdir_path):
            continue  # Skip if the folder doesn't exist
        for root, dirs, files in os.walk(subdir_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if not is_valid_image(file_path):
                    corrupted_files.append(file_path)
                    os.remove(file_path)
    return corrupted_files


data_directory = "./Data"
corrupted_files = clean_truncated_images(data_directory)
print("Removed files:", corrupted_files)
