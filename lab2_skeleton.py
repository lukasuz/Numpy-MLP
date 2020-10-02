from __future__ import print_function
import numpy as np

#In this first part, we just prepare our data (mnist) 
#for training and testing

#import keras
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).T
X_test = X_test.reshape(X_test.shape[0], num_pixels).T
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
X_train  = X_train / 255
X_test  = X_test / 255


#We want to have a binary classification: digit 0 is classified 1 and 
#all the other digits are classified 0

y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==0.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==0.0)[0]] = 1
y_test = y_new


y_train = y_train.T
y_test = y_test.T


m = X_train.shape[1] #number of examples

#Now, we shuffle the training set
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]


class Dense_Neural_Network():
    def __init__(self, architecture, loss="cross_entropy", learning_rate=0.001, weight_scale=0.01, rnd_seed=2020):
        self.learning_rate = learning_rate
        self.weight_scale = weight_scale
        self.rnd_seed = rnd_seed
        self.architecture = architecture
        self.num_layers = len(architecture)
        if loss == "cross_entropy":
            self.loss = self.cross_entropy_loss
        else:
            raise Exception("No other loss functions implemented yet.")

        # Network params
        self.weights = {}
        self.biases = {}
        self.activations = {}

        # Activations and deriviatives during training
        self.Z = {}
        self.A = {}
        self.dA = {}
        self.dZ = {}
        self.dW = {}
        self.db = {}
        self.delta = {}
        
        self.init_params()

    def init_params(self):
        """ Initializes the parameters

        # Argument:
            weigh_scale: float, scales the weights generated randomly by a normal distribution
        """
        np.random.seed(self.rnd_seed) # only relevant during initialization

        for idx, layer in enumerate(self.architecture):
            layer_idx = idx + 1
            layer_input_size = layer["input_dim"]
            layer_nodes = layer["nodes"]
            layer_activation = layer["activation"]

            self.weights[layer_idx] = np.random.randn(layer_input_size, layer_nodes) * self.weight_scale
            self.biases[layer_idx] = np.random.randn(layer_nodes, 1) * self.weight_scale
            if layer_activation == "relu":
                self.activations[layer_idx] = self.relu
            elif layer_activation == "sigmoid":
                self.activations[layer_idx] = self.sigmoid
            else:
                raise Exception("Invalid activation at layer {0}. Use relu or sigmoid instead.".format(layer_idx))

    def sigmoid(self, X, forward=True):
        if forward:
            return 1 / (1 + np.exp(-X))
        else:
            sig = self.sigmoid(X)
            return sig * (1 - sig)

    def relu(self, X, forward=True):
        if forward:
            return np.max(0, X)
        else:
            raise Exception("Relu backprob not implemented yet.")
            
    def forward_propagation(self, X):
        self.A[0] = X

        for i in range(1, self.num_layers + 1):
            # Access values
            activation = self.activations[i]
            W = self.weights[i]
            b = self.biases[i]
            prev_A = self.A[i-1]

            # Do layer calculations
            Z = np.matmul(prev_A, W) + b.T

            A = activation(Z)

            # Save intermediate values for backprop
            self.Z[i] = Z
            self.A[i] = A

        return self.A[i]

    def backward_propagation(self, Y, Y_hat):

        indx = self.num_layers
        self.dA[indx] = self.loss(Y, Y_hat, forward=False)

        for i in reversed(range(1, self.num_layers + 1)):
            m = self.dA[i].shape[0]
            activation = self.activations[i]

            self.dZ[i] = self.dA[i] * activation(self.Z[i], forward=False)
            self.dW[i] = (self.dZ[i].T @ self.A[i-1]) / m
            self.db[i] = np.sum(self.dZ[i], axis=0, keepdims=True) / m
            self.dA[i-1] = self.dZ[i] @ self.weights[i].T

    def update(self):
        for i in range(1, self.num_layers + 1):
            self.weights[i] -= self.learning_rate * self.dW[i].T
            self.biases[i] -= self.learning_rate * self.db[i].T

    def cross_entropy_loss(self, Y, Y_hat, forward=True):
        m = Y_hat.shape[0]

        if forward:
            return (-1/m) * (np.sum(np.log(Y_hat) * Y) + 
                        np.sum(np.log(1 - Y_hat) * (1 - Y))) 
        else:
            return - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    def mse_loss(self, Y, Y_hat, forward=True):
        if forward:
            return 1/2 * np.sum((Y - Y_hat)**2)
        else:
            return (Y - Y_hat)

    def train(self, X, Y, epochs, batch_size):
        losses = []
        for i in range(epochs):
            Y_hat = self.forward_propagation(X)
            loss = self.loss(Y, Y_hat)
            losses.append(loss)
            print(loss)

            self.backward_propagation(Y, Y_hat)
            self.update()


architecture_task_1 = [
    {"input_dim": 28*28, "nodes": 10, "activation": "sigmoid"},
    {"input_dim": 10, "nodes": 10, "activation": "sigmoid"},
    {"input_dim": 10, "nodes": 10, "activation": "sigmoid"},
    {"input_dim": 10, "nodes": 1, "activation": "sigmoid"}
]

NN = Dense_Neural_Network(architecture_task_1)
NN.train(X_train[:,:100].T, y_train[:,:100].T, 1000)

# #Display one image and corresponding label 
# import matplotlib
# import matplotlib.pyplot as plt
# i = 3
# print('y[{}]={}'.format(i, y_train[:,i]))
# plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
# plt.axis("off")
# plt.show()


#Let start our work: creating a neural network
#First, we just use a single neuron. 


#####TO COMPLETE

