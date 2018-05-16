import numpy as np

"""
Loads input/output of iris dataset
"""


def load_iris_data(path):
    f = open(path, "r")
    lines = f.readlines()
    samples = []
    iris2index = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2};
    for line in lines:
        split_line = line.split(",")
        if len(split_line) == 5:
            x = np.array([float(c) for c in split_line[:4]])
            x = x.reshape(4, 1)
            y = np.zeros((3, 1))
            y[iris2index[split_line[4].strip()], 0] = 1.0
            sample = (x, y)
            samples.append(np.array(sample))
    return np.array(samples)
