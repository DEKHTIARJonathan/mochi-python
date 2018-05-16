import cPickle as pickle

import net
import utils
from config import *
from data import load_reddit, load_mnist, load_digits, load_chars, load_iris_data
from net import optimizer, layer
import numpy as np


# layers1 = [
#     layer.Convolutional(input_dim=[1, 28, 28], num_patterns=32, filter_dim=[1, 5, 5],
#                         optimizer=optimizer.RMSProp(1e-2),
#                         activation=(utils.softplus, utils.softplus_prime)),
#     layer.MaxPool(input_dim=[32, 24, 24], pool_dim=[1, 2, 2]),
#     layer.Convolutional(input_dim=[32, 12, 12], num_patterns=64, filter_dim=[32, 3, 3],
#                         optimizer=optimizer.RMSProp(1e-2),
#                         activation=(utils.softplus, utils.softplus_prime)),
#     layer.MaxPool(input_dim=[64, 10, 10], pool_dim=[1, 2, 2]),
#     layer.VanillaFeedForward([64 * 5 * 5, 1], [1000, 1],
#                              optimizer=optimizer.RMSProp(5e-3),
#                              activation=(utils.softplus, utils.softplus_prime)),
#     layer.VanillaFeedForward([1000, 1], [10, 1],
#                              optimizer=optimizer.RMSProp(5e-3),
#                              activation=(utils.linear, utils.const_one))  # 13520
# ]


# layers = [
#     layer.VanillaFeedForward([784, 1], [300, 1],
#                              optimizer=optimizer.RMSProp(5e-3),
#                              activation=(utils.softplus, utils.softplus_prime)),
#     layer.VanillaFeedForward([300, 1], [10, 1],
#                              optimizer=optimizer.RMSProp(5e-3),
#                              activation=(utils.linear, utils.const_one))
# ]


# if train_from_file:
#     print("Training from File...")
#     with open(layers_file, "rb") as handle:
#         layers = pickle.load(handle)

# data_file = 'data/mnist.pkl.gz'
# training_data, validation_data, test_data = load_mnist(data_path='data/datasets/mnist.pkl.gz')
# feedforward_net = net.FeedForwardNet(layers, layers_file)
# feedforward_net.train(training_data, niter=20000, batch_size=50)

# result, domain_size, index_to_word, word_to_index = load_reddit(use_existing=True,
#                                                                 existing_path="data/datasets/training_data.pickle",
#                                                                 data_path="data/datasets/reddit_text.csv")

# shakespeare_char_data, domain_size, index_to_word, word_to_index = load_chars("data/datasets/shakespeare.txt")

layers = [
    layer.VanillaFeedForward([4, 1], [3, 1],
                             optimizer=optimizer.RMSProp(1e-1),
                             activation=(utils.linear, utils.const_one)),
    # layer.VanillaFeedForward([3, 1], [3, 1],
    #                          optimizer=optimizer.RMSProp(1e-1),
    #                          activation=(utils.so, utils.const_one))
]

iris_net = net.FeedForwardNet(layers, "dummy.pkl")
data = load_iris_data('data/datasets/iris-data.txt')
np.random.shuffle(data)
training_data = data[:100]
val_data = data[100:]
val_x = np.array([sample[0] for sample in val_data])
val_y = np.array([sample[1] for sample in val_data])
print(len(data))

iris_net.train(data, niter=5000, batch_size=1)

preds = iris_net.predict(val_x)
print(preds)
print(np.array([utils.onehot2Int(y) for y in val_y]))
num_correct = 0
for i in range(len(val_data)):
    if preds[i] == utils.onehot2Int(val_y[i]):
        num_correct += 1.0
print(num_correct/float(len(val_y)))



# recurrent_net = net.RecurrentNet(layers, "dummy.pkl")
# recurrent_net.train(shakespeare_char_data, index_to_word, niter=700000, len_seq=25)


# digits_data, domain_size, index_to_digit, digit_to_index = load_digits()
# recurrent_net = net.RecurrentNet(layers, "dummy.pkl")
# recurrent_net.train(digits_data, index_to_digit, niter=500000, len_seq=10)





# neural_net.validate(validation_data)

# for x, y in training_data[:10]:
#    neural_net.grad_check(x, y)
