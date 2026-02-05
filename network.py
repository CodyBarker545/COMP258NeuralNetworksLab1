import numpy as np
import random

# Neural Network class based on Michael Nielsen's "Neural Networks and Deep Learning"
class Network:
    def __init__(self, sizes):
        """
        Initialize the neural network.
        
        :param sizes: List indicating the number of neurons in each layer.
                      For example, [784, 30, 10] is a 3-layer network:
                      784 input neurons, 30 hidden neurons, 10 output neurons.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Initialize biases for all layers except input (random normal distribution)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Initialize weights connecting each layer (also using random normal distribution)
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Return the output of the network if 'a' is input.
        
        Applies the weighted input (z = w·a + b) followed by sigmoid activation.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the network using mini-batch stochastic gradient descent.
        
        :param training_data: List of tuples (x, y)
        :param epochs: Number of epochs to train
        :param mini_batch_size: Number of samples per mini-batch
        :param eta: Learning rate
        :param test_data: Optional test dataset for evaluation after each epoch
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)  # Shuffle the training data each epoch
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            #if test_data:
                #print(f"Epoch {j} : {self.evaluate(test_data)} / {n_test}")
            #else:
                #print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        """
        Update network weights and biases by applying gradient descent
        using backpropagation on a single mini-batch.
        """
        # Initialize gradient accumulators
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # Accumulate gradients
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Update weights and biases using the average gradient from the mini-batch
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Return a tuple (nabla_b, nabla_w) representing the gradient for the cost function
        with respect to each bias and weight in the network.
        """
        # Initialize gradient arrays
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # FORWARD PASS
        activation = x
        activations = [x]  # List to store all activations, layer by layer
        zs = []            # List to store all z vectors (weighted inputs)

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # BACKWARD PASS
        # Compute delta for output layer
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # Backpropagate through previous layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Evaluate network performance on test data.
        Returns the number of correct predictions.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(pred == label) for (pred, label) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Return the derivative of the cost function with respect to the output activations.
        (Here, using the quadratic cost: 0.5 * ||y - a||^2)
        """
        return output_activations - y

# Helper functions
def sigmoid(z):
    """The sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

