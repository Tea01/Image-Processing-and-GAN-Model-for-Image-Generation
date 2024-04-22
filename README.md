# Image-Processing-and-GAN-Model-for-Image-Generation

## Introduction
This project implements an image processing pipeline in Python using PyTorch, PIL, and torchvision. The main features include converting JPEG images to JPG format, filtering out unreadable images, and training a Generative Adversarial Network (GAN) to generate new images based on the processed dataset.

## Features
JPEG to JPG Conversion: Automatically renames all .jpeg files to .jpg within a specified directory, ensuring compatibility and standardization of file format.
Robust Image Loader: Custom dataset loader that ignores corrupted or unreadable image files, thus enhancing the robustness of the model training process.
Image Transformation: Implements several transformations for data augmentation including resizing, flipping, rotating, affine transformations, and color jitter.
Generative Adversarial Network: Utilizes a GAN composed of a Generator and a Discriminator to generate new images from noise.

## Dependencies
Python 3.x
PyTorch
torchvision
matplotlib
numpy
PIL
