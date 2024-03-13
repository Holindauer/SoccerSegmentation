import torch
import torch.nn as nn
from copy import deepcopy   

class EarlyStopping:
    """
    The EarlyStopping class implements a simple early stopping mechanism to stop 
    training if the test loss does not improve for a certain number of epochs.
    """
    def __init__(self, patience : int):

        # init early stopping params
        self.patience = patience
        self.counter = 0
        self.testLossMin = float('inf')

    def __call__(self, model : nn.Module, testLoss : float):

        # Save the model if test loss has decreased
        if testLoss <= self.testLossMin:
            
            # save model and update min test loss
            self.bestModel = deepcopy(model.state_dict())
            self.testLossMin = testLoss 

        # Test loss didn't improve but patience not exceeded
        elif testLoss > self.testLossMin and self.counter < self.patience:

            # increase patience counter
            self.counter += 1

        else: # patience exceeded
            
            # save the best model
            torch.save(self.bestModel, 'bestModel.pth')
            return True

        return False