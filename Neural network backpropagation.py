import numpy as np


#Weight matrix
fc1=np.load('ANN0.npy')
fc2=np.random.randn(128*10).reshape([128,10])

label=np.load('mnistLabel.npy')
input=np.load('mnist.npy')/255.0
input=np.reshape(input,[len(input),-1])

########################################Onehot Encode
D = np.identity(10)[label]

#---------------------------------------------End code

accuracy = []

for ep in range(100):
    
    
########################################Forward Pass X1=???
#first layer output
    x1 = (np.dot(fc1.T,input.T)).T
    #x1 = (np.dot(input,fc1)) 
    
    
#---------------------------------------------End code
    
    
    
    
########################################Forward Pass A1=???
# for relu
    A1 = x1.copy()
    A1[A1<0] = 0
    
#---------------------------------------------End code
    
    
    
    
########################################Forward Pass X2=???
#second layer
    x2 = (np.dot(A1,fc2))
    
#---------------------------------------------End code
    
    
    
    
########################################Forward Pass A2=???
#second layer output transpose by sigmoid
    A2 = 1/(1+np.exp(-x2))
    
#---------------------------------------------End code
    
    
    
########################################Backward delta=???
#sigmoid derivative*predict value -->delta(loss)
    backward_delta =A2*(1-A2)*(D-A2)

#---------------------------------------------End code
    
    
    
    
########################################Backward grad=???
    
    grad = np.dot(((-2)*A1).T,backward_delta)

#---------------------------------------------End code
    
    
    
    fc2=fc2-0.05*grad
    
    choose=np.argmax(A2,1)
    print(str(ep+1)+": "+str(np.sum(choose==label)/len(label)))
    accuracy.append(np.sum(choose == label) / len(label))

    
import matplotlib.pyplot as plt 

plt.plot(accuracy)
plt.title("Neural Network backpropagation")
plt.xlabel("Iterate")
plt.ylabel("Accuracy")
plt.show()






















