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

X, y = spiral_data(250, 2)
y = y.reshape(-1, 1)

# Show spiral data structure
# plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
# plt.show()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, w_regl1=0, w_regl2=0, b_regl1=0, b_regl2=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.w_regl1 = w_regl1
        self.w_regl2 = w_regl2
        self.b_regl1 = b_regl1
        self.b_regl2 = b_regl2
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
        
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        
        if self.w_regl1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.w_regl1 * dL1
            
        if self.w_regl2 > 0:
            self.dweights += 2 * self.w_regl2 * self.weights
            
        if self.b_regl1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.b_regl1 * dL1
            
        if self.b_regl2 > 0:
            self.dbiases += 2 * self.b_regl2 * self.biases
        
# Rectilinear Activation Function
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        
#Softmax Activation Function
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
        
#Sigmoid Activation Function
class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
        
class Loss:
    def calc(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
    def regularization_loss(self, layer):
        reg_loss = 0
        
        if layer.w_regl1 > 0:
            reg_loss += layer.w_regl1 * np.sum(np.abs(layer.weights))
            
        if layer.w_regl2 > 0:
            reg_loss += layer.w_regl2 * np.sum(layer.weights**2)

        if layer.b_regl1 > 0:
            reg_loss += layer.b_regl1 * np.sum(np.abs(layer.biases))
            
        if layer.b_regl2 > 0:
            reg_loss += layer.b_regl2 * np.sum(layer.biases**2)
            
        return reg_loss
    
#Categorical Crossentropy Loss Function
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
        
#Softmax-based Loss Function
class Loss_Softmax(Loss):
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
        
#Binary Categorical Crossentropy loss Function
class Loss_BCCE(Loss):
    def forward(self, y_hat, y):
        y_hat_clipped = np.clip(y_hat, 1e-7, 1 - 1e-7)
        
        sample_losses = -(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        
        return sample_losses
    
    def backward(self, dvalues, y):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        
        self.dinputs = -(y / clipped_dvalues - (1 - y) / (1 - clipped_dvalues)) / outputs
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

#Adaptive Gradient Optimizer
class AdaGrad:
    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
     
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        
        layer.weights += -self.current_lr * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_lr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        
    def post_updating(self):
        if self.decay:
            self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        self.iterations += 1
        
#Root Mean Squared Propagation Optimizer
class RMSProp:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
     
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
        
        layer.weights += -self.current_lr * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_lr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        
    def post_updating(self):
        if self.decay:
            self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        self.iterations += 1

#Adaptive Momentum Optimizer
class Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
     
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
            
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        layer.weights += -self.current_lr * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_lr * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
        
    def post_updating(self):
        if self.decay:
            self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        self.iterations += 1

        
dl1 = Layer_Dense(2, 64, w_regl2=5e-4, b_regl2=5e-4)
act1 = Activation_ReLU()

dl2 = Layer_Dense(64, 1)
act2 = Activation_Sigmoid()
loss_act = Loss_BCCE()

# optimizer = Stochastic_GD(learning_rate=1., decay=1e-3, momentum=0.9)
# optimizer = AdaGrad(learning_rate=1., decay=1e-3)
# optimizer = RMSProp(learning_rate=0.02, decay=1e-5, rho=0.999)
optimizer = Adam(learning_rate=0.05, decay=5e-7)

loss_history = []

for epoch in range(5001):
    dl1.forward(X)
    act1.forward(dl1.output)

    dl2.forward(act1.output)
    act2.forward(dl2.output)
    
    data_loss = loss_act.calc(act2.output, y)

    predictions = (act2.output > 0.5) * 1
    accuracy = np.mean(predictions == y)
    
    reg_loss = loss_act.regularization_loss(dl1) + loss_act.regularization_loss(dl2)
    loss = data_loss + reg_loss
    
    if not epoch % 100:
        print(f'epoch: {epoch}\nacc: {accuracy:.3f}\nloss: {loss:.3f}\ndata loss: {data_loss:.3f}\n')

    loss_act.backward(act2.output, y)
    
    act2.backward(loss_act.dinputs)
    dl2.backward(act2.dinputs)
    
    act1.backward(dl2.dinputs)
    dl1.backward(act1.dinputs)

    optimizer.update_params(dl1)
    optimizer.update_params(dl2)
    optimizer.post_updating()
    
    loss_history.append(loss)
    
plt.plot(loss_history)
plt.title('Loss over training')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

X_test, y_test = spiral_data(50, 2)
y_test = y_test.reshape(-1, 1)

dl1.forward(X_test)
act1.forward(dl1.output)
dl2.forward(act1.output)
act2.forward(dl2.output)
loss = loss_act.calc(act2.output, y_test)
predictions = (act2.output > 0.5) * 1
accuracy = np.mean(predictions == y_test)
print("d\nTESTING RESULTS:")
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')