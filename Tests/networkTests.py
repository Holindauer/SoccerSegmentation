from segmentationNet import SegmentationNet, SegNetConfig
import torch

def test_segmentation_net_output_shape():
    # Define the test configuration for SegmentationNet
    config = SegNetConfig(
        downsampling_channels=[64, 128, 256],
        upsampling_channels=[256, 128, 64],
        final_channels=[32, 16],
        num_classes=11
    )

    # Instantiate the network with the test configuration
    model = SegmentationNet(config)
    
    # Define the input shape (e.g., batch size of 1, 3 channels, 256x256 pixels)
    input_shape = (1, 3, 256, 256)
    # Expected output shape given the num_classes in config (e.g., batch size of 1, 11 classes, 256x256 pixels)
    expected_output_shape = (1, config.num_classes, 256, 256)
    
    # Create a dummy input tensor based on the input shape
    dummy_input = torch.rand(input_shape)
    
    # Pass the dummy input through the model
    output = model(dummy_input)
    
    # Assert that the output shape is as expected
    assert output.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, but got {output.shape}"
    
    print("Test passed: The model outputs the correct shape.")


# Run tests
if __name__ == "__main__":
    test_segmentation_net_output_shape()
