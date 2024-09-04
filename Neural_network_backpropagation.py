import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, fc1_path, fc2_shape, label_path, input_path):
        """
        初始化神經網絡，包括載入權重矩陣、數據和進行一熱編碼
        """
        # 載入第一層的權重矩陣
        self.fc1 = np.load(fc1_path)
        # 初始化第二層的權重矩陣，形狀為fc2_shape
        self.fc2 = np.random.randn(*fc2_shape).reshape(fc2_shape)
        
        # 載入標籤和輸入數據，並對輸入數據進行歸一化
        self.label = np.load(label_path)
        self.input = np.load(input_path) / 255.0
        # 將輸入數據展平
        self.input = np.reshape(self.input, [len(self.input), -1])

        # 將標籤進行一熱編碼
        self.D = np.identity(10)[self.label]
        
        # 用於儲存每個訓練周期的準確率
        self.accuracy = []
    
    def relu(self, x):
        """
        ReLU 激活函數
        """
        x[x < 0] = 0
        return x
    
    def sigmoid(self, x):
        """
        Sigmoid 激活函數
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        計算 Sigmoid 函數的導數
        """
        return x * (1 - x)
    
    def forward_pass(self):
        """
        執行前向傳播，計算每一層的輸出
        """
        # 計算第一層的輸出
        self.x1 = np.dot(self.input, self.fc1)
        # 應用 ReLU 激活函數
        self.A1 = self.relu(self.x1)

        # 計算第二層的輸出
        self.x2 = np.dot(self.A1, self.fc2)
        # 應用 Sigmoid 激活函數
        self.A2 = self.sigmoid(self.x2)
    
    def backward_pass(self):
        """
        執行反向傳播，計算梯度並更新權重
        """
        # 計算誤差項（delta）
        backward_delta = self.sigmoid_derivative(self.A2) * (self.D - self.A2)

        # 計算第二層的梯度
        grad = np.dot((-2) * self.A1.T, backward_delta)

        # 更新第二層的權重
        self.fc2 -= 0.05 * grad
    
    def train(self, epochs):
        """
        訓練神經網絡，進行多個訓練周期
        """
        for ep in range(epochs):
            # 前向傳播
            self.forward_pass()
            # 反向傳播
            self.backward_pass()
            # 預測結果
            choose = np.argmax(self.A2, axis=1)
            # 計算準確率
            accuracy = np.sum(choose == self.label) / len(self.label)
            # 儲存準確率
            self.accuracy.append(accuracy)
            # 輸出當前周期的準確率
            print(f"Epoch {ep+1}: Accuracy = {accuracy:.4f}")
    
    def plot_accuracy(self):
        """
        繪製訓練過程中的準確率變化圖
        """
        plt.plot(self.accuracy)
        plt.title("Neural Network Backpropagation")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.show()

# 使用範例
if __name__ == "__main__":
    # 初始化神經網絡，指定權重檔案和數據檔案的路徑
    nn = NeuralNetwork(
        fc1_path='ANN0.npy', 
        fc2_shape=(128, 10), 
        label_path='mnistLabel.npy',   
        input_path='mnist.npy'
    )
    # 開始訓練，設定訓練周期為100
    nn.train(epochs=100)
    # 繪製準確率變化圖
    nn.plot_accuracy()
