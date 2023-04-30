# Self-Driving Cars Percpetion - Image Classification
16-664 Self-Driving Cars: Perception & Control

Carnegie Mellon University

[In-class Kaggle Challenge](https://www.kaggle.com/competitions/16664-spring-2023-task-1-image-classification)

---
# Comparing Multiple Techniques for Vehicle Classification

![driving](https://user-images.githubusercontent.com/83327791/232208460-b2cf6115-ff77-4cf4-b3a8-8aae766edb36.gif)

## Abstract
*Vehicle classification is a critical task in machine learning and computer vision, with many real-world applications. In this project, we explore multiple vehicle classification techniques using pre-processing and deep learning models on driving scenes. We compare the performance of different approaches and demonstrate their potential for real-time vehicle classification. We hope our results contribute to advancing research in machine learning and computer vision and offer a starting point for improving accuracy in vehicle classification.*


## Introduction
Vehicle classification has been an active area of research in the fields of machine learning and computer vision. We explored four diffnt approaches for vehicle classification using 10,204 snapshots of scenes via various pre-processing methods for training data (RGB images of game scenes) and convolutional neural network architectures. The dataset included 7,573 training images and 2,631 test images, each containing an RGB image, a 3D bounding box, a camera matrix, and a label. The dataset had three unique labels corresponding to sets of similar vehicles. (e.g. sedans and SUVs are classified as the label 1).

## Bounding Boxes
Rotation vectors, centroids, sizes, and camera matrices of 3D bounding boxes are given for each of the vehicles in the training scene image.

- 3D bounding box

![image](https://user-images.githubusercontent.com/83327791/232202552-4068d83b-3812-4df2-950a-c101f867e0d3.png)
![image](https://user-images.githubusercontent.com/83327791/232202570-85c5ebab-c5f3-4664-8485-7e2bf6ad4adc.png)
![image](https://user-images.githubusercontent.com/83327791/232202583-3b6dcc6d-c75a-452f-8910-9798721e96b5.png)

- Cropping a vehicle from a snpashot using 2D bounding box (computed from max & min vertices of 3D bounding box)

![image](https://user-images.githubusercontent.com/83327791/232202875-7a2c5fa1-d6f6-4e48-b906-be7bb1c6b190.png)
![image](https://user-images.githubusercontent.com/83327791/232202880-8dfc639f-4fbc-4b2a-ab69-d91894d5b906.png)
![image](https://user-images.githubusercontent.com/83327791/232202885-c18edf86-1a75-4ebe-a726-060ed73ad585.png)
![image](https://user-images.githubusercontent.com/83327791/232202889-1c3fa432-f318-4969-a365-6ede30871b63.png)


## Experiments
The first method involved fine-tuning a pre-trained ResNet18 model by passing training images through its 17 convolutional layers, concatenating bounding box coordinates to feed the result into a fully connected layer.

The second method transferred a pre-trained ResNet18 and replaced its last layer with a dense layer having 3 output neurons. The model was fine-tuned with training images.

In the third method, we combined a simple neural network, a feature extractor, and a pre-trained ResNet18. Unlike the first method, we converted the 3D bounding box coordinates for each image to 2D coordinates and used them to crop vehicles from the training images. The feature extractor then used a pre-trained ResNet18 to extract features from the cropped images. These features were then fed into the simple neural network, consisting of 2 dense layers, to capture additional features. We replaced the last layer of another pre-trained ResNet18 with this simple neural network and fine-tuned it using the full-scale training images.

In the fourth method, we modified a pre-trained ResNet18 by replacing its ReLU activation function with a leaky ReLU with negative slope of 0.01, and its last fully connected layer with three dense layers. We added these extra layers to capture more complex features. We chose leaky ReLU based on empirical results reported in \cite{2015activation}, which showed that it can lower test loss for convolutional neural networks such as CIFAR-10 and CIFAR-100 showing that it can lower test loss for convolutional neural networks using CIFAR benchmark. We set the negative slope as 0.001 to experiment.

$$ y_i = \begin{cases}
x_i & \text{if } x_i \geq 0 \\
\frac{x_i}{a_i} & \text{if } x_i < 0
\end{cases} $$


## Results
Our four experimental approaches were evaluated using 2,631 test images extracted from the game's 3D universe. Method 1 achieved 50.5% accuracy, while Method 2, Method 3, and Method 4 achieved 69.1%, 59.1%, and 60.8% accuracy, respectively. However, these results are dependent on the choice of hyperparameters and may vary accordingly.


## Conclusion
Our second method achieved the highest test accuracy among all experimental methods suggesting that adding extra layers, activation functions, or data augmentation techniques does not always improve image classification performance. Experimenting with different combinations of these techniques may be necessary, and our findings provide additional insights for the field of autonomous vehicle perception.

## References
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. pages 770–778, 2016.

[2] B. Xu, N. Wang, T. Chen, and M. Li. Empirical evaluation of rectified activations in convolution network. 2015.


## Appendix
| Method | Description |
| --- | --- |
| Method 1 | <ul><li>Model - ResNet18 (17 convolutional layers & 1 dense layer; ReLU activation)</li><li>Data - Full-scale scene images</li><li>Additional Features - Coordinates of 3D bounding box around each vehicle</li><li>Preprocessing - Resizing, Normalization, Horizontal-Flipping</li><li>Stochastic Gradient Descent, Cross Entropy Loss</li></ul> |
| Method 2 | <ul><li>Model - ResNet18 (17 convolutional layers & 1 dense layer; ReLU activation); Last layer modified to accept 3 classes</li><li>Data - Full-scale scene images</li><li>Preprocessing - Resizing, Normalization</li><li>Adam Optimizer, Cross Entropy Loss</li></ul> |
| Method 3 | <ul><li>Model - ResNet18 (17 convolutional layers & 1 dense layer; ReLU activation); Feature Extractor (another ResNet18); Simple Neural Networks (2 dense layers)</li><li>Data - Full-scale scene images</li><li>Additional Features - Coordinates of 2D bounding box around each vehicle</li><li>Preprocessing - Resizing, Normalization, Horizontal Flipping, Rotation</li><li>Adam Optimizer, Cross Entropy Loss</li></ul> |
| Method 4 | <ul><li>Model - Modified ResNet18 (17 convolutional layers & 3 dense layers; Leaky ReLU activation)</li><li>Data - Full-scale scene images</li><li>Preprocessing - Resizing, Normalization</li><li>Adam Optimizer, Cross Entropy Loss</li></ul> |

