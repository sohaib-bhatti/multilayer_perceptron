import classifier as NN
import yaml
import numpy as np

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(message)s',
                    level=logging.WARNING)


def load_images(file_path):
    """Load image data from the IDX file format."""
    with open(file_path, 'rb') as f:
        # Read the header information
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        # Read the image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, num_rows, num_cols)
        data = data.reshape(num_images, num_rows * num_cols)

    return data


def load_labels(file_path):
    """Load label data from the IDX file format."""
    with open(file_path, 'rb') as f:
        # Read the header information
        magic_number = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')

        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        one_hot_encode = True

        if one_hot_encode is True:
            labels = np.eye(10)[labels]

    return labels


def main():
    # Load the training and testing data
    train_images = load_images('data/MNIST/raw/train-images-idx3-ubyte')
    train_labels = load_labels('data/MNIST/raw/train-labels-idx1-ubyte')
    test_images = load_images('data/MNIST/raw/t10k-images-idx3-ubyte')
    test_labels = load_labels('data/MNIST/raw/t10k-labels-idx1-ubyte')

    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    train_images = np.reshape(train_images, [60000, 28 * 28])
    test_images = np.reshape(test_images, [10000, 28 * 28])

    # pull hyperparameters from config file
    hyperparameters_file = 'hyperparameters'

    with open(f'{hyperparameters_file}.yaml', 'r') as file:
        hyperparameters = yaml.full_load(file)

    num_hidden_layers =\
        hyperparameters['hidden_layers']['num_hidden_layers']
    hidden_layer_width =\
        hyperparameters['hidden_layers']['hidden_layer_width']

    training_rate = hyperparameters['gradients']['training_rate']

    batch_size = hyperparameters['batch']['batch_size']
    num_epochs = hyperparameters['batch']['num_epochs']

    num_inputs = 28**2
    num_outputs = 10
    mlp = NN.MLP(num_inputs, num_outputs,
                 num_hidden_layers, hidden_layer_width,
                 NN.leaky_relu, NN.leaky_relu_prime,
                 NN.softmax, NN.softmax_prime,
                 NN.cross_entropy, NN.cross_entropy_prime, training_rate,
                 NN.he_init)

    NN.train_loop(train_images, train_labels, batch_size,
                  60000, num_epochs, mlp)

    accuracy = NN.test_loop(zip(test_images, test_labels), 10000, mlp)

    print(accuracy)


if __name__ == "__main__":
    main()
