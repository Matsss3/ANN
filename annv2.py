import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(0)

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

X, y = spiral_data(100, 3)

# Show spiral data structure
# plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
# plt.show()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
        
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        
class Activation_SoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
class Loss:
    def calc(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CCE(Loss):
    def forward(self, y_hat, y):
        samples = len(y_hat)
        y_hat_clipped = np.clip(y_hat, 1e-7, 1 - 1e-7)
        
        #Check for categorical labels
        if len(y.shape) == 1:
            correct_confidences = y_hat_clipped[range(samples), y]
        #Check for one-hot coded labels
        elif len(y.shape) == 2:
            correct_confidences = np.sum(y_hat_clipped * y, axis=1)
        
        loss = -np.log(correct_confidences)
        return loss
    
    def backward(self, dvalues, y):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y.shape) == 1:
            y = np.eye(labels)[y]

        self.dinputs = -y / dvalues
        self.dinputs = self.dinputs / samples
        
dl1 = Layer_Dense(2, 3)
act1 = Activation_ReLU()

dl2 = Layer_Dense(3, 3)
act2 = Activation_SoftMax()

loss_func = Loss_CCE()

dl1.forward(X)
act1.forward(dl1.output)

dl2.forward(act1.output)
act2.forward(dl2.output)

loss = loss_func.calc(act2.output, y)
predictions = np.argmax(act2.output, axis=1)

if len(y) == 2:
    y = np.argmax(y, axis=1)

accuracy = np.mean(predictions == y)

print(act2.output[:5])
print("Loss: ", loss)
print("Accuracy: ", accuracy)