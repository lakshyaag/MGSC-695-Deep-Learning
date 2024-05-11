# Assignment 1: Image Classification using Convolutional Neural Networks

## Objective

The aim of this assignment is to help you gain practical experience with Convolutional Neural Networks (CNNs), specifically in the context of image classification. You'll implement a CNN using a deep learning framework (like PyTorch or TensorFlow), and get familiar with key concepts such as convolutions, pooling, and fully connected layers.

## Dataset

CIFAR-10 or CIFAR-100. These datasets are widely used in the machine learning community and will allow you to compare your model's performance with reported benchmarks.

## Tasks

- Load and preprocess the dataset: The CIFAR datasets consist of 32x32 color images, split into training and testing sets. You might need to normalize the images and convert the labels to one-hot encoded vectors.
- Implement a CNN: Using the learned concepts, design a CNN architecture suitable for this task. Remember to consider aspects like the depth of the network, and the balance between model complexity and the risk of overfitting.
- Train your CNN: Train the network using the training part of the dataset. Make sure to implement mechanisms to monitor the training process, such as saving the model weights with the best validation accuracy.
- Evaluate your model: Evaluate your model on the test dataset. Provide a report on your model's performance using metrics such as accuracy, precision, recall, and F1 score.

## Success Criteria

The success of this assignment will be determined by:

- Correct implementation and training of a CNN using the chosen deep learning framework.
- The ability to achieve a reasonable accuracy on the test dataset - aim for accuracy comparable to simple baseline models reported in literature.
- Clear documentation of your decisions regarding the chosen architecture, training process, and evaluation results.
