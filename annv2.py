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
        
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        
        for i, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            self.dinputs[i] = np.dot(jacobian_matrix, single_dvalues)
        
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
        
class Loss_Softmax:
    def __init__(self):
        self.activation = Activation_SoftMax()
        self.loss = Loss_CCE()
        
    def forward(self, inputs, y):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calc(self.output, y)
    
    def backward(self, dvalues, y):
        samples = len(dvalues)
    
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y] -= 1
        self.dinputs = self.dinputs / samples

#Stochastic Gradient Descent Optimizer
class Stochastic_GD:
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
     
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            weight_updates = self.momentum * layer.weight_momentums - self.current_lr * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_lr * layer.dbiases
            layer.bias_momentums = bias_updates
        else:        
            layer.weights += -self.learning_rate * layer.dweights
            layer.biases += -self.learning_rate * layer.biases
            
        layer.weights += weight_updates
        layer.biases += bias_updates
        
    def post_updating(self):
        if self.decay:
            self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        self.iterations += 1

        
dl1 = Layer_Dense(2, 64)
act1 = Activation_ReLU()

dl2 = Layer_Dense(64, 3)
loss_soft = Loss_Softmax()

optimizer = Stochastic_GD(learning_rate=1., decay=1e-3, momentum=0.)

loss_history = []

for epoch in range(1001):
    dl1.forward(X)
    act1.forward(dl1.output)

    dl2.forward(act1.output)
    loss = loss_soft.forward(dl2.output, y)

    predictions = np.argmax(loss_soft.output, axis=1)

    if len(y) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)
    
    if not epoch % 50:
        print(f'epoch: {epoch}\nacc: {accuracy:.3f}\nloss: {loss:.3f}\n')

    loss_soft.backward(loss_soft.output, y)
    dl2.backward(loss_soft.dinputs)

    act1.backward(dl2.dinputs)
    dl1.backward(act1.dinputs)

    optimizer.update_params(dl1)
    optimizer.update_params(dl2)
    optimizer.post_updating()
    
    loss_history.append(loss)
    
plt.plot(loss_history)
plt.title('Loss over iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()