import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

class DataReader:
    """
    This class is responsible for loading, preparing, and plotting image data from the Data directory.
    """

    def __init__(self, imgSize=(256, 256)):
        # Compose transform to apply to images
        self.transform = transforms.Compose([   
            transforms.Resize(imgSize),  # Resize images to a fixed size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet values
        ])
        
    def loadDataset(self, dataPath: str):
        self.dataset = datasets.ImageFolder(root=dataPath, transform=self.transform)

    def imshow(self, img):
        img = img.clone()
        # Reverse the normalization
        for i in range(3):  # Assuming img is 'C x H x W'
            img[i] = img[i] * self.transform.transforms[-1].std[i] + self.transform.transforms[-1].mean[i]
        img = np.clip(img.numpy(), 0, 1)
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.axis('off')  # Don't show axes for images

    def plot_examples(self, num_images=4, class_name=None):
        plt.figure(figsize=(10, 10))
        plotted = 0
        for i, (img, label) in enumerate(self.dataset):
            if class_name is not None:
                # Skip images not belonging to the specified class
                if self.dataset.classes[label] != class_name:
                    continue
            
            ax = plt.subplot(1, num_images, plotted + 1)
            self.imshow(img)
            ax.set_title(f"Label: {self.dataset.classes[label]}")
            plt.tight_layout()
            plotted += 1
            if plotted == num_images:
                break

        if plotted == 0:
            print(f"No images found for class '{class_name}'.")

