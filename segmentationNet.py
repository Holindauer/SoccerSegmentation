from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


"""
segmentationNet.py contains a simple U-Net like architecture for semantic segmentation of images. 
"""

class ConvDownsamplingBlock(nn.Module):
    """
    ConvDownsamplingBlock is a simple block that applies a convolutional layer followed by batch normalization
    and a leaky ReLU activation function. This block is in the downsampling of the input.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvDownsamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding='same')
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        return x

class PoolingBlock(nn.Module):
    """
    PoolingBlock is a simple block that applies max pooling to the input, followed by dropout. This block is
    used after the convolutional layers to downsample the input.
    """
    def __init__(self, dropout_rate: float = 0.25):
        super(PoolingBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.dropout(x)
        return x

class UpsampleBlock(nn.Module):
    """
    UpsampleBlock is a simple block that applies upsampling to the input, followed by a convolutional layer.
    This block is used to upsample the input to match the original input size.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = ConvDownsamplingBlock(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        return x

@dataclass
class SegNetConfig:
    """
    The SegNetConfig dataclass that holds the configuration for the SegmentationNet architecture. 

    @param downsampling_channels is a list of integers representing the number of channels in each downsampling block.
    @param upsampling_channels is a list of integers representing the number of channels in each upsampling block.
    @param final_channels is a list of integers representing the number of channels in the final adjustment layers.
    @param num_classes is an integer representing the number of classes for the segmentation task.
    @param channel_multiplier is an integer representing the multiplier for the number of channels in each block.
    """
    downsampling_channels: List[int]
    upsampling_channels: List[int]
    final_channels: List[int]
    num_classes: int
    channel_multiplier: int = 4

class SegmentationNet(nn.Module):
    """
    SegmentationNet is a simple U-Net like architecture for semantic segmentation of images. The general idea
    behind this architecture is to pass the input through a series of convolutional layers to downsample the
    input, then upsample the output to match the input size. This is similar to an autoencoder, but with the
    output being a segmentation map instead of the original input.
    """
    def __init__(self, config: SegNetConfig):
        super().__init__()
        
        # module lists for the downsampling, pooling, and upsampling blocks
        self.down_blocks: nn.ModuleList = nn.ModuleList()
        self.pool_blocks: nn.ModuleList = nn.ModuleList()
        self.up_blocks: nn.ModuleList = nn.ModuleList()
        self.channel_multiplier: int = config.channel_multiplier
        
        in_channels: int = 3  # Assuming RGB input

        # Assemble downsampling blocks
        for out_channels in config.downsampling_channels:
            self.down_blocks.append(self._make_layers(ConvDownsamplingBlock, in_channels, [out_channels] * self.channel_multiplier))
            self.pool_blocks.append(PoolingBlock())
            in_channels = out_channels

        # Assemble upsampling blocks
        for out_channels in config.upsampling_channels:
            self.up_blocks.append(UpsampleBlock(in_channels, out_channels))
            in_channels = out_channels
        
        # Final adjustment to match desired output shape
        self.final_layers: nn.Sequential = self._make_layers(ConvDownsamplingBlock, in_channels, config.final_channels)
        self.output_conv: nn.Conv2d = nn.Conv2d(config.final_channels[-1], config.num_classes, kernel_size=2, stride=1, padding='same')

    def _make_layers(self, block: nn.Module, in_channels: int, out_channels_list: List[int]) -> nn.Sequential:

        # Assembles a series of layers
        layers: List[nn.Module] = []
        for out_channels in out_channels_list:
            layers.append(block(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Downsample
        for down_block, pool_block in zip(self.down_blocks, self.pool_blocks):
            x = down_block(x)
            x = pool_block(x)

        # Upsample
        for up_block in self.up_blocks:
            x = up_block(x)

        # Final adjustments
        x = self.final_layers(x)
        x = self.output_conv(x)
        return x
