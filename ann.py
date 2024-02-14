import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

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

X, y = spiral_data(100, 2)

X = (X - X.mean()) / X.std()

#Show dataset
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
# plt.title('Spiral Dataset')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    
def backward(Z1, A1, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = oneHotCode(Y)
    dZ2 = A2 - one_hot_Y.T
    dW2 = 1 / m * A1.T.dot(dZ2)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.dot(dZ2.T) * deriv_ReLU(Z1).T
    dW1 = 1 / m * dZ1.dot(X)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(weights1, biases1, weights2, biases2, dweights1, dbiases1, dweights2, dbiases2, alpha):
    weights1 -= alpha * dweights1.T
    biases1 -= alpha * dbiases1
    weights2 -= alpha * dweights2
    biases2 -= alpha * dbiases2

    return weights1, biases1, weights2, biases2

def deriv_ReLU(Z):
    return Z > 0

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

def oneHotCode(Y):
    one_hot_y = np.zeros((Y.size, Y.max() + 1))
    one_hot_y[np.arange(Y.size), Y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        self.output = exp_values / np.sum(exp_values, axis = 1, keepdims = True)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(sample), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

dense1 = Layer_Dense(2, 6)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(6, 2)
activation2 = Activation_Softmax()

def gradient_descent(X, y, iterations, alpha):
    weights1 = dense1.weights
    biases1 = dense1.biases
    weights2 = dense2.weights
    biases2 = dense2.biases
    
    loss_history = []

    for i in range(iterations):
        #Forwards
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        #Backwards
        DWeights1, DBiases1, DWeights2, DBiases2 = backward(dense1.output, activation1.output, activation2.output, dense2.weights, X, y)

        #Update
        weights1, biases1, weights2, biases2 = update_params(weights1, biases1, weights2, biases2, DWeights1, DBiases1, DWeights2, DBiases2, alpha)
        
        #Loss
        loss_function = Loss_CategoricalCrossentropy()
        loss = loss_function.calculate(activation2.output, y)
        
        loss_history.append(loss)
        
    #Results
    plt.plot(loss_history)
    plt.title('Loss over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    return weights1, biases1, weights2, biases2, np.argmax(activation2.output, axis=1)

dense1.weights, dense1.biases, dense2.weights, dense2.biases, predictions = gradient_descent(X, y, 100, 0.1)

# Example usage for testing
_, y_test = spiral_data(100, 2)

# Compare predicted labels with true labels
accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)
