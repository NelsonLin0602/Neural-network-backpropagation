# Neural Network Backpropagation in Python

## Overview

This project implements a simple neural network with a single hidden layer using Python and NumPy. The network is trained on the MNIST dataset for digit classification. The code demonstrates forward propagation, ReLU activation, sigmoid activation, and backpropagation to update weights. Accuracy is tracked and plotted over epochs to visualize the training process.

## Structure
- **NeuralNetwork Class:** 
  - Handles initialization, forward pass, backward pass, and training.  
    處理初始化、前向傳播、反向傳播和訓練。
  - Contains methods for ReLU and sigmoid activation functions, and their derivatives.  
    包含 ReLU 和 Sigmoid 激活函數及其導數的方法。

- **Methods:**
  - `__init__`: Initializes weight matrices, loads data, and sets up one-hot encoding.  
    初始化權重矩陣，載入資料並設置 one-hot 編碼。
  - `relu`: Applies ReLU activation function.  
    應用 ReLU 激活函數。
  - `sigmoid`: Applies sigmoid activation function.  
    應用 Sigmoid 激活函數。
  - `sigmoid_derivative`: Computes the derivative of the sigmoid function.  
    計算 Sigmoid 函數的導數。
  - `forward_pass`: Executes forward propagation through the network.  
    通過網絡執行前向傳播。
  - `backward_pass`: Computes loss and updates weights.  
    計算損失並更新權重。
  - `train`: Trains the network over a specified number of epochs.  
    在指定的訓練週期內訓練網絡。
  - `plot_accuracy`: Plots accuracy over training epochs.  
    繪製訓練週期內的準確率。

## Code Explanation

1. **NeuralNetwork Class**:
    - **Initialization (`__init__`)**:
        - Loads the weight matrix for the first layer (`fc1`) and initializes the weight matrix for the second layer (`fc2`).  
          載入第一層的權重矩陣 `fc1`，並初始化第二層的權重矩陣 `fc2`。
        - Loads label data and input data, normalizing the input data.  
          載入標籤數據和輸入數據，並對輸入數據進行歸一化處理。
        - Converts labels to one-hot encoding for easier computation.  
          將標籤轉換為 one-hot 編碼，以便於計算。

    - **Activation Functions**:
        - `relu`: ReLU activation function, sets negative values to 0.  
          ReLU 激活函數，將負值設置為 0。
        - `sigmoid`: Sigmoid activation function, compresses output to a range between 0 and 1.  
          Sigmoid 激活函數，將輸出壓縮到 0 到 1 之間。
        - `sigmoid_derivative`: Computes the derivative of the sigmoid function for backpropagation.  
          計算 Sigmoid 函數的導數，用於反向傳播。

    - **Forward Propagation (`forward_pass`)**:
        - Computes the output of the first layer and applies the ReLU activation function.  
          計算第一層的輸出並應用 ReLU 激活函數。
        - Computes the output of the second layer and applies the Sigmoid activation function to get the final predictions.  
          計算第二層的輸出並應用 Sigmoid 激活函數，得到最終的預測結果。

    - **Backward Propagation (`backward_pass`)**:
        - Computes the error term (delta), representing the difference between predicted and actual values.  
          計算誤差項（delta），表示預測值與實際值之間的差異。
        - Calculates the gradient for the second layer and updates the weight matrix `fc2` using a learning rate of 0.05.  
          計算第二層的梯度，並使用學習率 0.05 更新權重矩陣 `fc2`。

    - **Training (`train`)**:
        - Runs multiple training epochs, each consisting of forward and backward propagation.  
          執行多個訓練週期，每個週期包括前向傳播和反向傳播。
        - Calculates and stores accuracy for each epoch and prints the result.  
          計算並儲存每個週期的準確率，並將結果打印出來。

    - **Plotting Accuracy (`plot_accuracy`)**:
        - Plots the accuracy curve over the training epochs, making it easier to observe the model's learning process.  
          繪製訓練週期內的準確率曲線，便於觀察模型的學習過程。

2. **Usage Example**:
    - In the main part of the code, an instance of the `NeuralNetwork` class is created, with paths to the weight files and data files specified.  
      在程式的主體部分，創建 `NeuralNetwork` 類別的實例，並指定權重文件和數據文件的路徑。
    - The training process is executed with the number of epochs set to 100.  
      訓練過程設定訓練週期數為 100。
    - After training, the accuracy curve is plotted and displayed.  
      訓練完成後，繪製並顯示準確率曲線。

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
