import pickle
import numpy as np


class ConnectError(Exception):
    pass


class DataError(Exception):
    pass


def non_linear(data):
    if type(data) == np.ndarray:
        return (data > 0) * data
    else:
        return max([data, 0])


def d_non_linear(data):
    if type(data) == np.ndarray:
        return (data > 0) * 1


def inserting(n):
    temp = [0 for i in range(10)]
    temp[n] = 1
    return


def loss(expection, data, type_=0):
    if type_ == 0:
        loss = (expection - data) ** 2 / 2
    else:
        loss = (expection - data) ** 2
    return loss


def get_data():
    with open('mnist.pkl', 'rb') as f:
        data_set = pickle.load(f, encoding='iso-8859-1')
    data_train_list = [data_set[0][0][i].reshape((28, 28)).copy() for i in range(50000)]
    label_train_list = [np.array(inserting(j)) for j in data_set[0][1]]
    return data_train_list, label_train_list


class Layer:
    def __init__(self, node_num, front_num):
        self.node_num = node_num
        self.w = np.ones((node_num, front_num))
        self.b = np.ones((node_num, 1))
        self.type = 0

    def set_input_type(self):
        self.type = -1

    def set_output_type(self):
        self.type = 1

    def set_hidden_type(self):
        self.type = 0

    def back_connect(self, back_layer):
        if self != 1:
            self.back_layer = back_layer
        else:
            raise ConnectError('Last layer cannot back connect anything')

    def front_connect(self, front_layer):
        if self != -1:
            self.front_layer = front_layer
        else:
            raise ConnectError('First layer cannot front connect anything')

    def forward_prop(self):
        a = self.front_layer.output
        self.output = np.dot(self.w, a) + self.b

    def front_feed(self, data):
        if self.type == -1:
            self.output = np.dot(self.w, data) + self.b
        else:
            raise DataError('Only first layer can front feed')

    def get_expection(self):
        if self.type == 1:
            return self.output
        else:
            raise DataError('Only last layer can get expection')

    def back_feed(self, data):
        if self.type == 1:
            pass
        else:
            raise DataError('Only last layer can back feed')


class BP:
    def __init__(self, node_nums):
        self.node_nums = node_nums
        self.layers = []
        for i in range(1, len(node_nums)):
            temp = Layer(node_nums[i], node_nums[i - 1])
            if i == 1:
                temp.set_input_type()
            elif i == len(node_nums):
                temp.set_output_type()
            self.layers.append(temp)
            self.connect()

    def connect(self):
        for i in range(0, len(self.layers) - 1):
            self.layers[i].back_connect(self.layers[i + 1])
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].front_connect(self.layers[i - 1])

    def forward_prop(self, data):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].front_feed(data)
            else:
                self.layers[i].forward_prop()

    def get_expection(self):
        return self.layers[-1].get_expection()

    def get_loss(self, data):
        lossing = loss(self.get_expection(), data)
        return lossing

    def backward_prop(self, data):
        pass


a = get_data()
b = a[0][0].reshape((28 * 28, 1))
l = BP((784, 16, 10))
l.forward_prop(b)
