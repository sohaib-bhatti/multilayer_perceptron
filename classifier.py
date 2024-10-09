import numpy as np

from alive_progress import alive_bar

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='debug.log', filemode='w',
                    format='%(levelname)s: %(message)s',
                    level=logging.DEBUG)
# np.set_printoptions(precision=3)
# np.set_printoptions(suppress=True)


# a linear unit
class Linear:
    def __init__(self, num_inputs, num_outputs, initialization):
        stdev = initialization(num_inputs, num_outputs)
        self.w = np.random.normal(0, stdev, size=[num_inputs, num_outputs])
        self.b = np.zeros([1, num_outputs])

    def __call__(self, x):
        return x @ self.w + self.b


# a layer with an activation function
class DenseLayer:
    def __init__(self, num_inputs, num_outputs, activation, initalization):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation = activation
        self.linear = Linear(self.num_inputs, num_outputs, initalization)

    def __call__(self, x):
        z = self.linear(x)
        return z, self.activation(z)


# the neural network, compiles the DenseLayers
# performs forward and backward passes
# performs SGD
class MLP:
    def __init__(self, num_inputs, num_outputs,
                 num_hidden_layers, hidden_layer_width,
                 hidden_activation, hidden_activation_deriv,
                 output_activation, output_activation_deriv,
                 cost, cost_deriv, alpha, initialization):

        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.hidden_activation = hidden_activation
        self.hidden_activation_deriv = hidden_activation_deriv
        self.output_activation = output_activation
        self.output_activation_deriv = output_activation_deriv

        self.cost = cost
        self.cost_deriv = cost_deriv

        self.alpha = alpha

        self.num_layers = self.num_hidden_layers + 2

        self.layers = [DenseLayer(num_inputs, self.hidden_layer_width,
                                  hidden_activation, initialization)]

        for i in range(self.num_hidden_layers):
            self.layers.append(DenseLayer(self.hidden_layer_width,
                                          self.hidden_layer_width,
                                          hidden_activation, initialization))

        self.layers.append(DenseLayer(self.hidden_layer_width,
                                      self.num_outputs,
                                      output_activation, initialization))
        self.pre_act = [None] * self.num_layers
        self.post_act = [None] * self.num_layers

    # go through all layers
    # storing both the pre-activation and post-activation values
    def forward(self, x):
        z = x
        for i in range(self.num_layers):
            z, a = self.layers[i](z)
            self.pre_act[i] = z
            self.post_act[i] = a
        return a

    # backwards pass, figure out gradients
    def backward(self, label):
        dcdw = [np.zeros(w.linear.w.shape) for w in self.layers]
        dcdb = [np.zeros(b.linear.b.shape) for b in self.layers]

        delta = self.cost_deriv(label, self.post_act[-1])
        dcdw[-1] = np.dot(self.post_act[-2].T, delta)
        dcdb[-1] = delta[-1]

        for i in range(2, self.num_layers):
            delta = delta = np.dot(delta, self.layers[-i + 1].linear.w.T) *\
                self.hidden_activation_deriv(self.pre_act[-i])
            dcdw[-i] = np.dot(self.post_act[-i-1].T, delta)
            dcdb[-i] = np.sum(delta, axis=0, keepdims=True)
            loss = self.cost(label, self.post_act[-1])

        return dcdw, dcdb, loss

    # stochastic gradient descent
    # take minibatches and do backwards passes with them
    # change weights and biases accordingly
    def SGD(self, x_batch, y_batch):
        weight_change_w = [np.zeros(w.linear.w.shape) for w in self.layers]
        weight_change_b = [np.zeros(b.linear.b.shape) for b in self.layers]

        total_loss = 0
        for x, label in zip(x_batch, y_batch):
            self.forward(x)
            dcdw, dcdb, loss = self.backward(label)
            total_loss += loss  # Accumulate total loss
            for i in range(self.num_layers):
                weight_change_w[i] += dcdw[i]
                weight_change_b[i] += dcdb[i]

        # Apply the averaged weight updates
        for i in range(self.num_layers):
            self.layers[i].linear.w -= self.alpha * (weight_change_w[i] /
                                                     len(x_batch))
            self.layers[i].linear.b -= self.alpha * (weight_change_b[i] /
                                                     len(x_batch))

        # Log the average loss for this batch
        average_loss = total_loss / len(x_batch)
        return average_loss


# take MLP as argument and divide data into minibatches
# perform SGD for every minibatch and adjust MLP accordingly
# do this for multiple epochs
def train_loop(x, y, batch_size, set_size, num_epochs, mlp: MLP):
    num_batches = int(set_size/batch_size)
    for i in range(num_epochs):
        indices = np.random.choice(set_size, [num_batches, batch_size],
                                   replace=False)
        with alive_bar(total=num_batches) as bar:
            for j in range(num_batches):
                x_batch = np.take(x, indices[j], axis=0)
                y_batch = np.take(y, indices[j], axis=0)
                loss = mlp.SGD(x_batch, y_batch)
                bar()
            logging.debug(f"Average loss: {loss}")
    return mlp


# check MLP results against test data
def test_loop(testing_data, num_samples, mlp: MLP):
    num_correct = 0
    print("testing accuracy...")
    with alive_bar(total=num_samples) as bar:
        for sample, label in testing_data:
            pred = mlp.forward(sample)
            if np.argmax(pred) == np.argmax(label):
                num_correct += 1
            bar()

    accuracy = num_correct/num_samples * 100

    return accuracy


# weight initialization functions
def xavier_init(num_inputs, num_outputs):
    stdev = np.sqrt(2 / (num_inputs + num_outputs))
    return stdev


def he_init(num_inputs, num_outputs):
    stdev = np.sqrt(2 / num_inputs)
    return stdev


# activation functions
def ReLU(x):
    return x * (x > 0)


def ReLU_prime(x):
    return 1 * (x > 0)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_prime(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1/(1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def softmax_prime(x):
    softmax_output = softmax(x)
    s = softmax_output.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


# lost functions
def brier(label, pred):
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    return np.sum(np.square(label - pred))


def brier_prime(label, pred):
    return (label - pred)


def cross_entropy(label, pred):
    # logging.debug(f"label passed into loss: {label}")
    # logging.debug(f"pred passed into loss: {pred}")
    epsilon = 1e-15
    pred = np.clip(pred, epsilon, 1 - epsilon)
    return -np.sum(label * np.log(pred) + (1 - label) * np.log(1 - pred))


def cross_entropy_prime(label, pred):
    return pred - label
