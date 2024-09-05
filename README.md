# Neural Network Backpropagation in Python

## Overview

This project implements a simple neural network with a single hidden layer using Python and NumPy. The network is trained on the MNIST dataset for digit classification. The code demonstrates forward propagation, ReLU activation, sigmoid activation, and backpropagation to update weights. Accuracy is tracked and plotted over epochs to visualize the training process.

## Structure
- **NeuralNetwork Class:** 
  - Handles initialization, forward pass, backward pass, and training.  
  - Contains methods for ReLU and sigmoid activation functions, and their derivatives.  

- **Methods:**
  - `__init__`: Initializes weight matrices, loads data, and sets up one-hot encoding.  
  - `relu`: Applies ReLU activation function.  
  - `sigmoid`: Applies sigmoid activation function.  
  - `sigmoid_derivative`: Computes the derivative of the sigmoid function.  
  - `forward_pass`: Executes forward propagation through the network.  
  - `backward_pass`: Computes loss and updates weights.  
  - `train`: Trains the network over a specified number of epochs.  
  - `plot_accuracy`: Plots accuracy over training epochs.  

## Code Explanation

1. **NeuralNetwork Class**:
    - **Initialization (`__init__`)**:
        - Loads the weight matrix for the first layer (`fc1`) and initializes the weight matrix for the second layer (`fc2`).  
        - Loads label data and input data, normalizing the input data.  
        - Converts labels to one-hot encoding for easier computation.  

    - **Activation Functions**:
        - `relu`: ReLU activation function, sets negative values to 0.  
        - `sigmoid`: Sigmoid activation function, compresses output to a range between 0 and 1.  
        - `sigmoid_derivative`: Computes the derivative of the sigmoid function for backpropagation.  

    - **Forward Propagation (`forward_pass`)**:
        - Computes the output of the first layer and applies the ReLU activation function.  
        - Computes the output of the second layer and applies the Sigmoid activation function to get the final predictions.  

    - **Backward Propagation (`backward_pass`)**:
        - Computes the error term (delta), representing the difference between predicted and actual values.  
        - Calculates the gradient for the second layer and updates the weight matrix `fc2` using a learning rate of 0.05.  

    - **Training (`train`)**:
        - Runs multiple training epochs, each consisting of forward and backward propagation.  
        - Calculates and stores accuracy for each epoch and prints the result.  

    - **Plotting Accuracy (`plot_accuracy`)**:
        - Plots the accuracy curve over the training epochs, making it easier to observe the model's learning process.  

2. **Usage Example**:
    - In the main part of the code, an instance of the `NeuralNetwork` class is created, with paths to the weight files and data files specified.  
    - The training process is executed with the number of epochs set to 100.  
    - After training, the accuracy curve is plotted and displayed.  

## Usage

### Loading MNIST Data

Before running the code, you need to ensure that the MNIST data is properly loaded and saved as `.npy` files. Here’s how you can do it:

1. **Download and Preprocess MNIST Data:**

    If you haven't yet saved the MNIST data as `.npy` files, you can use the following code to do so:

    ```python
    from tensorflow.keras.datasets import mnist
    import numpy as np

    # Load MNIST data from Keras
    (x_train, y_train), (_, _) = mnist.load_data()

    # Save the images and labels as .npy files
    np.save('mnist.npy', x_train)
    np.save('mnistLabel.npy', y_train)
    ```

2. **Loading Data in the Neural Network:**

    Once you have the `.npy` files (`mnist.npy` and `mnistLabel.npy`) saved, you can load them in your code as shown in the `NeuralNetwork` class:

    ```python
    # Example of loading the data
    images = np.load('mnist.npy')
    labels = np.load('mnistLabel.npy')
    ```

3. **Instantiate and Train the Neural Network:**

    With the `.npy` files in place, you can instantiate and train the neural network:

    ```python
    nn = NeuralNetwork(fc1_path='ANN0.npy', fc2_shape=(128, 10), label_path='mnistLabel.npy', input_path='mnist.npy')
    nn.train(epochs=100)
    nn.plot_accuracy()
    ```

### Requirements

- Python 3.x
- NumPy
- Matplotlib
- TensorFlow (for downloading MNIST dataset)

Install the required packages using pip:

```bash
pip install numpy matplotlib tensorflow
