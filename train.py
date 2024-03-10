import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass

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
        
        # unpack config
        self.model = config.model
        self.trainLoader = config.trainLoader
        self.testLoader = config.testLoader
        self.criterion = config.criterion
        self.optimizer = config.optimizer
        self.epochs = config.epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def train(self):

        for epoch in range(self.epochs):

            # Set the model to training mode
            self.model.train()

            trainLoss = 0.0

            for i, data in enumerate(self.trainLoader, 0):

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
                trainLoss += loss.item()

                print(f"Batch {i+1} of {len(self.trainLoader)}: Step Loss {loss.item()}", end='\r')

            # test 
            testLoss = self.test()

            trainLoss /= len(self.trainLoader) / self.trainLoader.batch_size

            print(" " * 100, end='\r') # Clear the line
            print(f"Epoch {epoch+1} Train Loss: {trainLoss:.4f} Test Loss: {testLoss:.4f}")

        
        return self.model
    

    def test(self) -> float:

        # Set the model to evaluation mode
        self.model.eval()

        testLoss = 0.0

        for i, data in enumerate(self.testLoader, 0):

            # send the inputs and labels to the device
            inImages, segMasks = data[0].to(self.device), data[1].to(self.device)

            # Forward pass
            outputs = self.model(inImages)

            loss = self.criterion(outputs, segMasks)

            # update test metrics
            testLoss += loss.item()

        # return averaged loss
        return testLoss / len(self.testLoader) / self.testLoader.batch_size




