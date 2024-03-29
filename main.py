import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageDraw
import random

# np.random.seed(0)

# def spiral_data(points, classes):
#     X = np.zeros((points*classes, 2))
#     y = np.zeros(points*classes, dtype='uint8')
#     for class_number in range(classes):
#         ix = range(points*class_number, points*(class_number+1))
#         r = np.linspace(0.0, 1, points)  # radius
#         t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
#         X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
#         y[ix] = class_number
#     return X, y

# X, y = spiral_data(250, 2)
# y = y.reshape(-1, 1)

# Show spiral data structure
# plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
# plt.show()

def load_mnist_dataset(dataset, path):
    data = pd.read_csv(os.path.join(path, dataset) + '.csv')

    y = data['label'].values

    X = data.drop('label', axis=1).values

    return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path)
    return X, y

def rotate_imgs(images, angle):
    for image in images:
        img = Image.fromarray(image.reshape(28,28))
        rotated_img = img.rotate(random.randint(-(angle), angle))
        image = np.array(rotated_img)
        image.flatten()

    return images

def translate_imgs(images):
    shift = (random.randint(-5, 5), random.randint(-5, 5))
    for image in images:
        trans_img = Image.fromarray(image.reshape(28,28))
        image = trans_img.transform(trans_img.size, Image.AFFINE, (1, 0, shift[0], 0, 1, shift[1]))
    
    return images

def preprocessing(images):
    return translate_imgs(rotate_imgs(images, 9))

X, y = create_data_mnist('./mnist_dataset')

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)

X = X[keys]
y = y[keys]

X = X.reshape(X.shape[0], -1).astype(np.float32)

preprocessing(X)

X = (X - 127.5) / 127.5

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
        
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        
        return data_loss
    
    def calc_accumulated(self):
        data_loss = self.accumulated_sum / self.accumulated_count
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
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
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
        
class Drawing_Panel:
    def __init__(self, master):
        self.master = master
        self.master.title("Drawing Panel")
        
        self.canvas = Canvas(self.master, width=400, height=400, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.save_button = Button(self.master, text="Save", command=self.save_img)
        self.save_button.pack(side=tk.BOTTOM)
        
        self.old_x = None
        self.old_y = None
        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.image = Image.new("RGB", (400, 400), "white")
        self.draw = ImageDraw.Draw(self.image)
        
    def paint(self, event):
        x, y = event.x, event.y
        r = 15  
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="black")
        
    def save_img(self):
        temp_path = 'temp.png'
        self.image.save(temp_path)
        print(f'Updating image at: {temp_path}')
        
def main_drawing():
    root = tk.Tk()
    app = Drawing_Panel(root)
    root.mainloop()
    
def pre_process_image(URL):
    image_data = cv2.imread(URL, cv2.IMREAD_GRAYSCALE)
    image_data = cv2.resize(image_data, (28, 28))
    image_data = 255 - image_data
    image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
    
    return image_data
        
dense_layer_1 = Layer_Dense(X.shape[1], 64, w_regl2=5e-4, b_regl2=5e-4)
activation_1 = Activation_ReLU()

dense_layer_2 = Layer_Dense(64, 64)
activation_2 = Activation_ReLU()

dense_layer_3 = Layer_Dense(64, 10)
activation_3 = Activation_SoftMax()
loss_activation = Loss_CCE()

# optimizer = Stochastic_GD(learning_rate=1., decay=1e-3, momentum=0.9)
# optimizer = AdaGrad(learning_rate=1., decay=1e-3)
# optimizer = RMSProp(learning_rate=0.02, decay=1e-5, rho=0.999)
optimizer = Adam(learning_rate=0.001, decay=5e-7)

def train(X, y, dl1, dl2, dl3, act1, act2, act3, loss_act, optimizer, epochs=1, batch_size=None, print_every=1):
    train_steps = 1
    loss_history = []
        
    if batch_size is not None:
        train_steps = len(X) // batch_size
        
        if train_steps * batch_size < len(X):
            train_steps += 1

    loss_act.new_pass()
    
    for epoch in range(epochs+1):
        if batch_size is not None:
            batch_X = X
            batch_y = y
        else:
            batch_X = X[epoch*batch_size:(epoch + 1)*batch_size]
            batch_y = y[epoch*batch_size:(epoch + 1)*batch_size]
                    
        dl1.forward(batch_X)
        act1.forward(dl1.output)

        dl2.forward(act1.output)
        act2.forward(dl2.output)

        dl3.forward(act2.output)
        act3.forward(dl3.output)
        
        data_loss = loss_act.calc(act3.output, batch_y)

        predictions = np.argmax(act3.output, axis=1)
        if len(batch_y) == 2:
            batch_y = np.argmax(batch_y, axis=1)
        accuracy = np.mean(predictions == batch_y)
        
        reg_loss = loss_act.regularization_loss(dl1) + loss_act.regularization_loss(dl2) + loss_act.regularization_loss(dl3)
        loss = data_loss + reg_loss 

        loss_act.backward(act3.output, batch_y)
        
        act3.backward(loss_act.dinputs)
        dl3.backward(act3.dinputs)

        act2.backward(dl3.dinputs)
        dl2.backward(act2.dinputs)
        
        act1.backward(dl2.dinputs)
        dl1.backward(act1.dinputs)

        optimizer.update_params(dl1)
        optimizer.update_params(dl2)
        optimizer.update_params(dl3)
        optimizer.post_updating()
        
        if not epoch % print_every or epoch == train_steps -1:
            print(f'epoch: {epoch}\nacc: {accuracy:.3f}\nloss: {loss:.3f}\n')
            
        loss_history.append(loss)
        
    plt.plot(loss_history)
    plt.title('Loss over training')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
        
train(X=X, y=y, dl1=dense_layer_1, dl2=dense_layer_2, dl3=dense_layer_3, act1=activation_1, act2=activation_2, act3=activation_3, loss_act=loss_activation, optimizer=optimizer, epochs=500, batch_size=10, print_every=50)

while True:
    opt = input("Seguir? (Y,N): ")
    
    if opt.lower() != 'y':
        break
    
    #Open Drawing Panel
    main_drawing()

    #Preprocess Image
    final_img = pre_process_image('temp.png')

    dense_layer_1.forward(final_img)
    activation_1.forward(dense_layer_1.output)

    dense_layer_2.forward(activation_1.output)
    activation_2.forward(dense_layer_2.output)

    dense_layer_3.forward(activation_2.output)
    activation_3.forward(dense_layer_3.output)

    output = np.vstack(activation_3.output)

    prediction = np.argmax(output, axis=1)[0]

    print(f'================================\nEsto es: {prediction}')