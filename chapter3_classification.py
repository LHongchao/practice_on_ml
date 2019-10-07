import struct, os
import pandas as pd
from numpy import *
import numpy as np
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

def load_train_from_mnist(path):
    labels_path = os.path.join(path, 'train-labels.idx1-ubyte')
    images_path = os.path.join(path, 'train-images.idx3-ubyte')
    lbpath = open(labels_path, 'rb')
    magic, n = struct.unpack('>II', lbpath.read(8))
    labels = np.fromfile(lbpath, dtype=np.uint8)
    imgpath = open(images_path, 'rb')
    magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
    images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

def load_test_from_mnist(path):
    labels_path = os.path.join(path, 't10k-labels.idx1-ubyte')
    images_path = os.path.join(path, 't10k-images.idx3-ubyte')
    lbpath = open(labels_path, 'rb')
    magic, n = struct.unpack('>II', lbpath.read(8))
    labels = np.fromfile(lbpath, dtype=np.uint8)
    imgpath = open(images_path, 'rb')
    magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
    images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

def plot_pictures(images, rows=6, cols=6):
    fig, ax = plt.subplots(nrows=rows, ncols=cols, sharex='all', sharey='all')
    ax = ax.flatten()
    for i in range(rows * cols):
        img = images[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')
    plt.tight_layout()
    plt.show()




path = 'F:/data/mnist/'
X_train, y_train = load_train_from_mnist(path)
X_test,  y_test = load_test_from_mnist(path)
shuffle_index = np.random.permutation(60000)

X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 =(y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

result = sgd_clf.predict([3])
print(result)


