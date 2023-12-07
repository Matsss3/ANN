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

def update_params_adam(weights, biases, dweights, dbiases, m_w, v_w, m_b, v_b, t, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # m_w = (beta1 * m_w).T + (1 - beta1) * dweights
    # v_w = (beta2 * v_w).T + (1 - beta2) * (dweights ** 2)
    m_b = beta1 * m_b + (1 - beta1) * dbiases
    v_b = beta2 * v_b + (1 - beta2) * (dbiases ** 2)
    
    

    # weights -= alpha * m_w / (np.sqrt(v_w) + epsilon)
    biases -= alpha * m_b / (np.sqrt(v_b) + epsilon)

    return weights, biases, m_w, v_w, m_b, v_b

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

def gradient_descent(X, y, iterations, alpha, optimizer="adam"):
    weights1 = dense1.weights
    biases1 = dense1.biases
    weights2 = dense2.weights
    biases2 = dense2.biases
    
    m_w1, v_w1, m_b1, v_b1 = np.zeros_like(weights1), np.zeros_like(weights1), np.zeros_like(biases1), np.zeros_like(biases1)
    m_w2, v_w2, m_b2, v_b2 = np.zeros_like(weights2), np.zeros_like(weights2), np.zeros_like(biases2), np.zeros_like(biases2)
    
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
        if optimizer == 'adam':
            weights1, biases1, m_w1, v_w1, m_b1, v_b1 = update_params_adam(
                weights1, biases1, DWeights1, DBiases1, m_w1, v_w1, m_b1, v_b1, i, alpha=alpha
            )
            weights2, biases2, m_w2, v_w2, m_b2, v_b2 = update_params_adam(
                weights2, biases2, DWeights2, DBiases2, m_w2, v_w2, m_b2, v_b2, i, alpha=alpha
            )
        else:
            weights1, biases1, weights2, biases2 = update_params(weights1, biases1, weights2, biases2, DWeights1, DBiases1, DWeights2, DBiases2, alpha)
        

        #Loss
        loss_function = Loss_CategoricalCrossentropy()
        loss = loss_function.calculate(activation2.output, y)
        
        loss_history.append(loss)
        
    #Results
    # plt.plot(loss_history)
    # plt.title('Loss over iterations')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.show()

    return weights1, biases1, weights2, biases2

dense1.weights, dense1.biases, dense2.weights, dense2.biases = gradient_descent(X, y, 50, 5)