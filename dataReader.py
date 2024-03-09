import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List
import os
from PIL import Image



class Plot:
    """
    The Plot class contains methods for visualizing images and their labels. Plot is instantiated 
    within the DataReader class and provided the .plotExamples() wrapper for access to Plot metods.
    """
    def __init__(
            self, 
            mean: np.ndarray = np.array([0.485, 0.456, 0.406]),  # <-- from imagenet
            std: np.ndarray = np.array([0.229, 0.224, 0.225])
            ) -> None:
        
        # normalization parameters
        self.mean = mean
        self.std = std

    def unnormalize(self, img: torch.Tensor) -> torch.Tensor:
        """Reverses the normalization for visualization."""
        for t, m, s in zip(img, self.mean, self.std):
            t.mul_(s).add_(m)  # Multiply by std and add mean for each channel
        return img

    def imshow(self, img: torch.Tensor, unnormalize: bool = True) -> None:
        """Plots an image, un-normalizing if specified."""
        if unnormalize and img.shape[0] == 3:  # Check for 3 channels (RGB)
            img = self.unnormalize(img)
        img = img.detach().cpu()
        img = np.transpose(img.numpy(), (1, 2, 0))
        # Handle single-channel images
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        plt.imshow(img)
        plt.axis('off')  # Don't show axes for images

    def plot_examples(self, dataloader: DataLoader, num_pairs: int = 2, unnormalize: bool = True) -> None:
        """Plots pairs of original images and their segmentation masks."""
        plt.figure(figsize=(10, 5 * num_pairs))
        for images, masks in dataloader:
            for i in range(min(num_pairs, len(images))):
                plt.subplot(num_pairs, 2, 2*i + 1)
                self.imshow(images[i], unnormalize=unnormalize)
                plt.title("Original Image")
                
                plt.subplot(num_pairs, 2, 2*i + 2)
                self.imshow(masks[i], unnormalize=False)  # Masks don't need unnormalization
            break  # Display only the first batch
        plt.tight_layout()
        plt.show()


class SegmentationDataset(Dataset):
    """
    The SegmentationDataset class is a custom dataset class for loading original images and their corresponding
    segmentation masks. The class inherits from the PyTorch Dataset class and overrides the __getitem__ and __len__
    methods to load and return the images and masks.
    """
    def __init__(
            self, 
            original_dataset_path: str, 
            segmentation_dataset_path: str, 
            transform: transforms.Compose = None, 
            target_transform: transforms.Compose = None
            ):
        
        # Store the paths and transforms
        self.original_dataset_path = original_dataset_path
        self.segmentation_dataset_path = segmentation_dataset_path
        
        # seperate transform for segmentation masks vs og images
        self.transform = transform
        self.target_transform = target_transform 

        # Get the list of images and masks
        self.original_images = sorted(os.listdir(original_dataset_path))
        self.segmentation_masks = sorted(os.listdir(segmentation_dataset_path))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        # Load the original image and segmentation mask
        original_img_path = os.path.join(self.original_dataset_path, self.original_images[idx])
        segmentation_mask_path = os.path.join(self.segmentation_dataset_path, self.segmentation_masks[idx])

        # Open the images
        original_img = Image.open(original_img_path).convert("RGB")
        segmentation_mask = Image.open(segmentation_mask_path).convert("L")  # Assuming masks are grayscale

        # Apply the transforms
        if self.transform:
            original_img = self.transform(original_img)
        if self.target_transform:
            segmentation_mask = self.target_transform(segmentation_mask)

        return original_img, segmentation_mask

    def __len__(self) -> int:
        return len(self.original_images)

class DataReader:
    """
    The DataReader class provides an interface for loading, preparing, and
    visualizing datasets for use in training a semantic segmentation model.
    """
    def __init__(self, imgSize: Tuple[int, int] = (256, 256)):    
        # Original images transform
        self.transform = transforms.Compose([   
            transforms.Resize(imgSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Segmentation masks transform
        self.target_transform = transforms.Compose([
            transforms.Resize(imgSize),  # Resize to the same size as the original images
            transforms.ToTensor(),
        ])
        
        # Instantiate the Plot class
        self.plot = Plot()

    def loadSegmentationDataset(self, originals_path: str, segmentations_path: str):
        """Loads a dataset consisting of original images and their corresponding segmentation masks."""
        self.dataset = SegmentationDataset(
            originals_path, 
            segmentations_path, 
            transform=self.transform, 
            target_transform=self.target_transform
            )
        
    def wrapDataLoader(self, batchSize: int) -> DataLoader:
        """ Wraps the loaded segmentation dataset in a DataLoader. """
        return DataLoader(self.dataset, batch_size=batchSize, shuffle=True)
    
    def plotExamples(self, numImages: int, className: str) -> None:
        """Wrapper for Plot.plot_examples() using dataset and transform from the DataReader instance."""
        self.plot.plot_examples(self.dataset, self.transform, numImages, className)

