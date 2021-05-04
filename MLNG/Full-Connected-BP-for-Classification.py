import numpy as np
from matplotlib import pyplot as plt
import pickle
import tqdm
import scipy.io as sio
import os

class NN:
    def __init__(self, input_scale, out_put_scale):
        self.input_scale = input_scale
        self.output_scale = out_put_scale

        # Layer nums: hidden layers and output layer
        self.layer_nums = 0
        self.weights = []
        self.biases = []

    def __str__(self):
        scales = [i.shape for i in self.weights]
        return str(scales)

    def set_inout(self, input_scale, out_put_scale):
        self.input_scale = input_scale
        self.output_scale = out_put_scale
    
    def set_layers(self, *layer_scales):
        prev_dim = self.input_scale
        for scale in layer_scales:
            # weights change dim from prev one into next one 
            # by doing right-side matrix multiplication
            self.weights.append(np.random.randn(prev_dim, scale))
            self.biases.append(np.random.randn(1, scale))
            prev_dim = scale
        self.weights.append(np.random.randn(prev_dim, self.output_scale))
        self.biases.append(np.random.randn(1, self.output_scale))
        self.layer_nums = len(layer_scales) + 1
    
    def set_lr(self, lr):
        self.lr = lr

    def predict(self, input_data):
        # output class index for each sample
        data = input_data.copy()
        for weight, bias in zip(self.weights, self.biases):
            data = sigmoid(data @ weight + bias) > 0.5
        return np.argmax(data, axis=1)

    def train(self, input_data, output_data):
        # input data with dims: (samples_num, features_num), (samples_num, onehot_length)
        # output onehot loss: (samples_num, onehot_length)
        """
        a : output(sigmoided if needed) of layers ; layer 0 as input
        z : aw + b
        """
        input_data = input_data.copy()
        output_data = output_data.copy()
        a = [input_data]
        z = [0]

        for i in range(self.layer_nums):
            # forward propaganda
            # z[i+1] = a[i] @ w[i] + b[i] 
            # a[i+1] = sigmoid(z[i+1]) 
            z.append(a[i] @ self.weights[i] + self.biases[i])
            a.append(sigmoid(z[-1]))

        # compute loss
        loss = ((a[-1] - output_data) ** 2).sum()
        
        # back propaganda
        d = (a[-1] - output_data) # dl_da

        for i in range(self.layer_nums - 1, -1, -1):
            # dl_dz = dl_da * da_dz
            d *= dsigmoid(z[i + 1])

            # dw = dl_dz * dz_da
            # left multiplication with a.T
            dw = a[i].T @ d

            # db = dl_dz * 1
            db = d.copy()

            # dl_da = dl_dz * dz_da
            # right multiplication with w.T
            d = d @ self.weights[i].T

            # gradient descend
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db.mean(0)
        return loss

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def prepare_data():
    def prepare(item_set):
        X, y = item_set
        Y = np.eye(10)[y]
        return (X, Y, y)
    with open('mnist.pkl', 'rb') as f:
        data_set = pickle.load(f, encoding='iso-8859-1')
    ret = []
    for item in data_set:
        ret.append(prepare(item))
    return ret

if __name__ == '__main__':
    (train_x, train_ohy, train_y), (valid_x, valid_ohy, valid_y), (test_x, test_ohy, test_y) = prepare_data()
    fn = input('load pre-trained:').strip()
    epoch = 2000
    best_acc = 0
    try:
        si, myNet, train_loss, valid_accuracy = pickle.load(open(fn, 'rb'))
    except:
        myNet =  NN(input_scale=784, out_put_scale=10)
        myNet.set_lr(0.0001)
        myNet.set_layers(256, 64)
        train_loss = []
        valid_accuracy = []
        si = 0
    plt.ion()
    for i in tqdm.tqdm(range(si, epoch)):
        train_loss.append(myNet.train(train_x, train_ohy))
        valid_accuracy.append((myNet.predict(valid_x) == valid_y).sum() / len(valid_y) * 100)
        plt.subplot(1, 2, 1)
        plt.plot(list(range(len(train_loss))), train_loss, color='blue')
        plt.subplot(1, 2, 2)
        plt.plot(list(range(len(valid_accuracy))), valid_accuracy, color='red')
        plt.pause(0.001)
        with open('nn.pkl', 'wb') as f:
            pickle.dump([i, myNet, train_loss, valid_accuracy], f)
        if best_acc < valid_accuracy[-1]:
            best_acc = valid_accuracy[-1]
            with open('best.pkl', 'wb') as f:
                pickle.dump([i, myNet, train_loss, valid_accuracy], f)
    plt.ioff()

    plt.plot(list(range(len(train_loss))), train_loss, color='blue')
    plt.plot(list(range(len(valid_accuracy))), valid_accuracy, color='red')
    plt.show()

    print('valid:', valid_accuracy[-1])
    print('test:', (myNet.predict(test_x) == test_y).sum() / len(test_y) * 100)
