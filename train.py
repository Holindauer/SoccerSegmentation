import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from earlyStopping import EarlyStopping
from collections import OrderedDict

@dataclass
class TrainConfig:
    model: nn.Module
    trainLoader: DataLoader
    testLoader: DataLoader
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    epochs: int
    doEarlyStopping : bool
    esPatience : int

class Trainer:
    """
    The Trainer Class contains the training loop for a semantic segmentation model.
    """

    def __init__(self, config: TrainConfig):
        
        self.trainLoader = config.trainLoader
        self.testLoader = config.testLoader
        self.criterion = config.criterion
        self.optimizer = config.optimizer
        self.epochs = config.epochs

        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = config.model.to(self.device)
        print("Training on:", self.device)

        # init early stopping
        self.doEarlyStopping = config.doEarlyStopping
        if self.doEarlyStopping:
            self.earlyStopping = EarlyStopping(config.esPatience)
    
    def train(self) -> OrderedDict: # <-- model state dict

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

            # update metrics
            testLoss = self.test()
            trainLoss /= len(self.trainLoader) / self.trainLoader.batch_size

            # Early stopping
            if self.doEarlyStopping:
                if self.earlyStopping(self.model, testLoss):
                    print(f"Early stopping at epoch {epoch+1} with test loss {testLoss:.4f}")
                    break
            
            # print epoch metrics
            print(" " * 100, end='\r') # Clear the line
            print(f"Epoch {epoch+1} Train Loss: {trainLoss:.4f} Test Loss: {testLoss:.4f}")

        
        return self.earlyStopping.bestModel
    

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




