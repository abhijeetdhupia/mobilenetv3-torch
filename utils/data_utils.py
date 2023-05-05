from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Lambda
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import os
import torchvision.transforms as transforms


def compute_mean_std(data_dir, n_channels=3):
    """
    Computes the mean and standard deviation for each channel in the dataset.

    Args:
    data_dir (str): Path to the dataset directory.
    n_channels (int): Number of channels in the images. Default is 3 (RGB).

    Returns:
    tuple: (mean, std) of the dataset's channels.
    """
    pixel_sum = np.zeros(n_channels)
    pixel_squared_sum = np.zeros(n_channels)
    num_pixels = 0

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = np.array(Image.open(os.path.join(root, file)))
                num_pixels += np.prod(image.shape[:2])
                pixel_sum += np.sum(image, axis=(0, 1))
                pixel_squared_sum += np.sum(np.square(image), axis=(0, 1))

    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_squared_sum / num_pixels - np.square(mean))

    return mean / 255.0, std / 255.0


# Function to get the data transformations for training, validation, and test sets.
def get_data_transforms():
    """
    Returns data transforms for training, validation, and test sets.
    """
    
    # Define your imgaug augmenters here
    augmenters = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(scale=(0.8, 1.2), rotate=(-20, 20)),
        iaa.CropAndPad(percent=(-0.2, 0.2)),
        # ... add more augmenters as needed
    ])

    # Wrap the imgaug augmenters in a torchvision Lambda transform
    imgaug_transform = Lambda(lambda x: ia.imresize_single_image(augmenters.augment_image(np.array(x)), (224, 224)))

    data_transforms = {
        'train': transforms.Compose([
            imgaug_transform,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms


# Function to create data loaders for training, validation, and test sets.
def get_data_loaders(data_dir, data_transforms, batch_size, num_workers=0):
    """
    Returns data loaders for training, validation, and test sets.
    
    Args:
    data_dir (str): Path to the dataset directory containing 'train', 'val', and 'test' subdirectories.
    batch_size (int): Batch size for data loading.
    num_workers (int, optional): Number of subprocesses to use for data loading. Default is 0 (main process).
    """
    data_transforms = data_transforms
    
    # Create datasets for training, validation, and test sets using ImageFolder.
    image_datasets = {x: ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    
    # Create data loaders for training, validation, and test sets.
    data_loaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val', 'test']}

    return data_loaders