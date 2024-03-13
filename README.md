# SoccerSegmentation

This repository contains an implementation and training of the U-Net nueral network architecture for task of semantic segmentation of images of soccer players. 

Semantic segmentation is a classification task for image data, images are masked into different classes at the pixel level. Target examples have the pixels of their classes colored differently than the rest of the image. The goal is to pass an original image into the model and pass out a mask of the same size with the pixels colored according to their class.


<div style="text-align: center;">
    <img src="Data/originalTest/Frame%201%20%20(4).jpg" alt="original" width="500" /> 
    <img src="Data/segmentationTest/Frame%201%20%20(4).jpg___fuse.png" alt="original" width="500" /> 
</div>


# Model Description

In this repository I implemented a U-Net like model for this task. U-Net is a neural network architecture that is used for image segmentation. It is similar to an autoencoder, with there being a downsampling and upsampling component. The input image is compressed into a smaller representation and then expanded back to the original size using convolutional layers. This was implemented in the model.py file in native pytorch.

# Training

The model was trained in an AWS sagemaker notebook instances w/ cuda using MSE for loss. To see the training and results, take a look at [trainNotebook.ipynb](trainNotebook.ipynb)



