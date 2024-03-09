import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from copy import deepcopy

@dataclass
class TrainConfig:
    model: nn.Module
    trainLoader: DataLoader
    testLoader: DataLoader
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    epochs: int

class Trainer:
    """
    The Trainer Class contains the training loop for a semantic segmentation model.
    """

    def __init__(self, config: TrainConfig):
        
        # Set the model, data loaders, loss function, optimizer, and number of epochs
        self.model = config.model
        self.train_loader = config.trainLoader
        self.test_loader = config.testLoader
        self.criterion = config.criterion
        self.optimizer = config.optimizer
        self.epochs = config.epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def train(self):

        for epoch in range(self.epochs):

            # Set the model to training mode
            self.model.train()

            epochLoss = 0.0

            for i, data in enumerate(self.train_loader, 0):

                # send the inputs and labels to the device
                inImages, segMasks = data[0].to(self.device), data[1].to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inImages)

                loss = self.criterion(outputs, segMasks)

                # backprop and gradient descent
                loss.backward()
                self.optimizer.step()

                # update training metrics
                epochLoss += loss.item()

                print(f"Batch {i+1} of {len(self.train_loader)}: Step Loss {loss.item()}", end='\r')



            epochLoss /= len(self.train_loader) / self.train_loader.batch_size


            print(f"Epoch {epoch+1} Loss: {epochLoss:.4f}")

        
        return self.model
