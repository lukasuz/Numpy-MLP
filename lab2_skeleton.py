from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def one_hot_encode_labels(y, y_dim=None):
    y_int = y.astype(int).T
    if y_dim is None:
        y_dim = y_int.max() + 1

    Y = np.zeros((y_int.size, y_dim))
    Y[np.arange(y_int.size), y_int] = 1

    return Y

def binarize_labels(y, label):
    y_new = np.zeros(y.shape)
    y_new[np.where(y==label)[0]] = 1

    return y_new

def save_plot(values, labels, x_axis, y_axis, title):
    fig = plt.figure()
    for i in range(len(values)):
        plt.plot(values[i], label=labels[i])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.title(title)
    plt.savefig("./{0}.png".format(title))
    plt.close()

class Dense_Neural_Network():
    def __init__(self, architecture, loss="binary_cross_entropy", learning_rate=0.01, weight_scale=0.01, rnd_seed=2020):
        self.learning_rate = learning_rate
        self.weight_scale = weight_scale
        self.rnd_seed = rnd_seed
        self.architecture = architecture
        self.num_layers = len(architecture)

        if loss == "binary_cross_entropy":
            self.loss = self.binary_cross_entropy_loss
        elif loss == "softmax_cross_entropy":
            self.loss = self.softmax_cross_entropy_loss
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

        # Initialize parameters
        self.init_params()

    def init_params(self):
        np.random.seed(self.rnd_seed) # only relevant during initialization

        for idx, layer in enumerate(self.architecture):
            layer_idx = idx + 1
            layer_input_size = layer["input_dim"]
            layer_nodes = layer["nodes"]
            layer_activation = layer["activation"]

            self.weights[layer_idx] = np.random.randn(layer_input_size, layer_nodes) * self.weight_scale
            self.biases[layer_idx] = np.random.randn(layer_nodes, 1) * self.weight_scale
            if layer_activation == "sigmoid":
                self.activations[layer_idx] = self.sigmoid
            elif layer_activation == "softmax":
                self.activations[layer_idx] = self.softmax
            else:
                raise Exception("Invalid activation at layer {0}. Use sigmoid or softmax instead.".format(layer_idx))
    
    def sigmoid(self, X, forward=True):
        if forward:
            return 1 / (1 + np.exp(-X))
        else:
            sig = self.sigmoid(X)
            return sig * (1 - sig)
    
    def softmax(self, X, forward=True):
        if forward:
            X -= np.max(X) # So we avoid devision by zero and exploding values
            return (np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True))
        else:
            # gradient is gandled in softmax cross_entropy loss function
            return 1

    def forward_propagation(self, X, train=True):
        # temporary variables for saving intermediate outputs
        temp_A = {}
        temp_Z = {}
        temp_A[0] = X

        for i in range(1, self.num_layers + 1):
            # Access layer parameters and do calculations
            activation = self.activations[i]
            temp_Z[i] = temp_A[i-1] @ self.weights[i] + self.biases[i].T
            temp_A[i] = activation(temp_Z[i])

        # Save activations if in training mode
        if train:
            self.Z = temp_Z
            self.A = temp_A

        return temp_A[i]

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

    def binary_cross_entropy_loss(self, Y, Y_hat, forward=True):
        m = Y_hat.shape[0]

        if forward:
            return (-1/m) * (np.sum(np.log(Y_hat) * Y) + 
                        np.sum(np.log(1 - Y_hat) * (1 - Y))) 
        else:
            return - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    def softmax_cross_entropy_loss(self, Y, Y_hat, forward=True):
        m = Y_hat.shape[0]

        if forward:
            return (-1/m) * np.sum(np.log(Y_hat + 1e-8) * Y)
        else:
            return Y_hat - Y

    def accuracy(self, Y, Y_hat):
        m = Y.shape[0]
        if Y_hat.shape[1] > 1: # Find max activation and one hot encode
            Y_hat = np.argmax(Y_hat, axis=1)
            Y_hat = one_hot_encode_labels(Y_hat, y_dim=10)
            return 1/m * np.sum(Y * np.round(Y_hat))
        else:
            return 1/m * np.sum(Y == np.round(Y_hat))

    def validate(self, X, Y):
        Y_hat = self.forward_propagation(X, train=False)
        loss = self.loss(Y, Y_hat)
        acc = self.accuracy(Y, Y_hat)

        return loss, acc

    def train(self, X, Y, epochs, batch_size, val_split = 0.2, verbose=1):
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        # Shuffle incoming data
        num_samples = X.shape[0]
        shuffle_mask = np.arange(num_samples)
        np.random.shuffle(shuffle_mask)
        X_shuffled = X[shuffle_mask, :]
        Y_shuffled = Y[shuffle_mask, :]

        # Split into training and validation set
        num_samples_train = int(num_samples * (1 - val_split))
        X_train = X_shuffled[:num_samples_train, :]
        Y_train = Y_shuffled[:num_samples_train, :]

        X_val = X_shuffled[num_samples_train:, :]
        Y_val = Y_shuffled[num_samples_train:, :]

        batches = num_samples_train // batch_size

        for e in range(1, epochs + 1):
            if verbose:
                print("Epoch:", e)

            # Create an index mask for the current batch
            batch_indx_masks = np.arange(num_samples_train)
            np.random.shuffle(batch_indx_masks)
            
            for b in range(1, batches + 1):
                # Retrieve indices for current batch
                batch_indx_mask = batch_indx_masks[(b-1) * batch_size: b * batch_size]
                batch_X = X_train[batch_indx_mask]
                batch_Y = Y_train[batch_indx_mask]

                # Do forward propagation on current batch
                Y_hat = self.forward_propagation(batch_X, train=True)
                train_loss = self.loss(batch_Y, Y_hat)
                train_acc = self.accuracy(batch_Y, Y_hat)
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                
                # Do forward propagation on current batch
                Y_hat_val = self.forward_propagation(X_val, train=False)
                val_loss = self.loss(Y_val, Y_hat_val)
                val_acc = self.accuracy(Y_val, Y_hat_val)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                # Do backpropagation based on training set activations and update weights
                self.backward_propagation(batch_Y, Y_hat)
                self.update()

                # Logging
                if verbose:
                    percentage = int(b / batches * 100)
                    print("  Batch percentage: {0:d}%, Train loss: {1:f}, Validation loss: {2:f}, Train acc: {3:f}, Validation acc: {4:f}"
                        .format(percentage, train_loss, val_loss, train_acc, val_acc), end="\r")
                
            if verbose:
                print("")

        return train_losses, val_losses, train_accs, val_accs

if __name__ == "__main__":

    # Prepare data set according to lab intructions
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

    X_train = X_train.T
    X_test = X_test.T
    y_train = y_train.T
    y_test = y_test.T
    
    # Create a binarzied (detection of zeros) and one hot encoded version
    # of the data set
    y_train_bin = binarize_labels(y_train.T, 0.0)
    y_test_bin = binarize_labels(y_test.T, 0.0)

    Y_train_one_hot = one_hot_encode_labels(y_train.T)
    Y_test_one_hot = one_hot_encode_labels(y_test.T)

    batch_size = 10000
    epochs = 3
    verbose = 0

    plot_labels = ["Train", "Validation"]
    x_axis = "Step (batch-wise)"

    ### TASK 1, binary single layer perceptron
    print("Task 1")
    architecture_task_1 = [
        {"input_dim": 28*28, "nodes": 1, "activation": "sigmoid"}
    ]
    SLP = Dense_Neural_Network(architecture_task_1, learning_rate=0.09)
    train_losses, val_losses, train_accs, val_accs = SLP.train(X_train, y_train_bin, epochs, batch_size, verbose=verbose)
    test_loss, test_acc = SLP.validate(X_test, y_test_bin)
    save_plot([train_losses, val_losses], plot_labels, x_axis, "Loss", "Loss Task 1")
    save_plot([train_accs, val_accs], plot_labels, x_axis, "Accuracy", "Accuracy Task 1")
    print("Final loss: {0:f} and accuracy: {1:f}\n\n".format(test_loss, test_acc))

    ### TASK 2, binary MLP
    print("Task 2")
    architecture_task_2 = [
        {"input_dim": 28*28, "nodes": 64, "activation": "sigmoid"},
        {"input_dim": 64, "nodes": 1, "activation": "sigmoid"}
    ]
    MLP = Dense_Neural_Network(architecture_task_2, learning_rate=0.09)
    train_losses, val_losses, train_accs, val_accs = MLP.train(X_train, y_train_bin, epochs, batch_size, verbose=verbose)
    test_loss, test_acc = MLP.validate(X_test, y_test_bin)
    save_plot([train_losses, val_losses], plot_labels, x_axis, "Loss", "Loss Task 2")
    save_plot([train_accs, val_accs], plot_labels, x_axis, "Accuracy", "Accuracy Task 2")
    print("Final loss: {0:f} and accuracy: {1:f}\n\n".format(test_loss.T, test_acc))

    ### TASK 2 optional, do binary classification for all numbers
    for i in range(1, 10):
        print("Task 2 optional for number:", i)
        y_train_bin = binarize_labels(y_train.T, float(i))
        y_test_bin = binarize_labels(y_test.T, float(i))

        MLP = Dense_Neural_Network(architecture_task_2, learning_rate=0.09)
        train_losses, val_losses, train_accs, val_accs = MLP.train(X_train, y_train_bin, epochs, batch_size, verbose=verbose)
        test_loss, test_acc = MLP.validate(X_test, y_test_bin)
        print("Final loss: {0:f} and accuracy: {1:f}\n".format(test_loss, test_acc))

    ### TASK 3, multiclass MLP
    # need significantly bigger learning rate
    print("Task 3")
    architecture_task_3 = [
        {"input_dim": 28*28, "nodes": 64, "activation": "sigmoid"},
        {"input_dim": 64, "nodes": 10, "activation": "softmax"}
    ]
    multi_MLP = Dense_Neural_Network(architecture_task_3, loss="softmax_cross_entropy", learning_rate=0.9)
    train_losses, val_losses, train_accs, val_accs = multi_MLP.train(X_train, Y_train_one_hot, epochs, batch_size, verbose=verbose)
    test_loss, test_acc = multi_MLP.validate(X_test, Y_test_one_hot)
    save_plot([train_losses, val_losses], plot_labels, x_axis, "Loss", "Loss Task 3")
    save_plot([train_accs, val_accs], plot_labels, x_axis, "Accuracy", "Accuracy Task 3")
    print("Final loss: {0:f} and accuracy: {1:f}\n".format(test_loss, test_acc))

    ### Extra Task, deeper multiclass MLP
    print("Extra:")
    architecture_extra = [
        {"input_dim": 28*28, "nodes": 64, "activation": "sigmoid"},
        {"input_dim": 64, "nodes": 32, "activation": "sigmoid"},
        {"input_dim": 32, "nodes": 10, "activation": "softmax"}
    ]
    multi_MLP = Dense_Neural_Network(architecture_extra, loss="softmax_cross_entropy", learning_rate=0.9)
    train_losses, val_losses, train_accs, val_accs = multi_MLP.train(X_train, Y_train_one_hot, epochs, batch_size, verbose=verbose)
    test_loss, test_acc = multi_MLP.validate(X_test, Y_test_one_hot)
    save_plot([train_losses, val_losses], plot_labels, x_axis, "Loss", "Loss Task Extra")
    save_plot([train_accs, val_accs], plot_labels, x_axis, "Accuracy", "Accuracy Task Extra")
    print("Final loss: {0:f} and accuracy: {1:f}\n".format(test_loss, test_acc))
