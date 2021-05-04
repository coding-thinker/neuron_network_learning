import pickle
import numpy as np


class ConnectError(Exception):
    pass


class DataError(Exception):
    pass


def non_linear(data):
    return np.exp(data)/(np.exp(data) + 1)
    

def d_non_linear(data):
    return non_linear(data) * (1 - non_linear(data))


def inserting(n):
    temp = [0 for i in range(10)]
    temp[n] = 1
    return temp


def loss(expection, data, type_=0):
    if type_ == 0:
        loss = (expection - data) ** 2 / 2
    else:
        loss = (expection - data) ** 2
    return loss


def get_data():
    with open('MLNG/mnist.pkl', 'rb') as f:
        data_set = pickle.load(f, encoding='iso-8859-1')
    data_train_list = [data_set[0][0][i].reshape((28, 28)).copy() for i in range(50000)]
    label_train_list = [np.array(inserting(j)) for j in data_set[0][1]]
    return data_train_list, label_train_list


class Layer:
    def __init__(self, node_num, front_num):
        self.node_num = node_num
        self.w = np.zeros((node_num, front_num))
        self.b = np.zeros((node_num, 1))
        self.type = 0

    def set_learning_rate(self, lr):
        self.learning_rate = lr

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
        self.a = self.front_layer.output
        self.z = np.dot(self.w, self.a) + self.b
        self.output = non_linear(self.z)

    def front_feed(self, data):
        if self.type == -1:
            self.a = data
            self.z = np.dot(self.w, self.a) + self.b
            self.output = non_linear(self.z)
        else:
            raise DataError('Only first layer can front feed')

    def get_expection(self):
        if self.type == 1:
            return self.output
        else:
            raise DataError('Only last layer can get expection')

    def back_feed(self, data):
        if self.type == 1:
            t = (self.output - data)
            self.passing_delta = (self.output - data) * d_non_linear(self.z)
            self.gradient_w = np.dot(self.passing_delta, self.a.T)
            self.gradient_b = self.passing_delta.copy()
        else:
            raise DataError('Only last layer can back feed')

    def backward_prop(self):
        iterative_loss = self.back_layer.passing_delta
        self.passing_delta = np.dot(self.back_layer.w.T, iterative_loss) * d_non_linear(self.z)
        self.gradient_w = np.dot(self.passing_delta, self.a.T)
        self.gradient_b = self.passing_delta.copy()

    def update(self):
        self.w -= self.gradient_w * self.learning_rate
        self.b -= self.gradient_b * self.learning_rate


class BP:
    def __init__(self, node_nums):
        self.node_nums = node_nums
        self.layers = []
        for i in range(1, len(node_nums)):
            temp = Layer(node_nums[i], node_nums[i - 1])
            if i == 1:
                temp.set_input_type()
            elif i == len(node_nums) - 1:
                temp.set_output_type()
            self.layers.append(temp)
            self.connect()

    def set_learning_rate(self, lr):
        for each in self.layers:
            each.set_learning_rate(lr)

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

    def predict(self,data):
        self.forward_prop(data)
        return self.layers[-1].get_expection()

    def get_loss(self, data):
        lossing = loss(self.get_expection(), data)
        return lossing

    def backward_prop(self, data):
        for i in range(len(self.layers) - 1, -1, -1):
            if i == len(self.layers) - 1:
                self.layers[i].back_feed(data)
            else:
                self.layers[i].backward_prop()
        for each in self.layers:
            each.update()

    def train(self, x_data, y_data, rounds=1):
        for _ in range(rounds):
            self.forward_prop(x_data)
            self.backward_prop(y_data)
        return self.get_loss(y_data)


def main():
    a = get_data()
    l = BP((784, 16, 10))
    l.set_learning_rate(0.01)
    for i in range(2500):
        b = a[0][i].reshape((28 * 28, 1))
        c = a[1][i].reshape((10, 1))
        l.train(b, c, 1000)
        if i % 100 == 0:
            print(str(i/25000*100) + '\t' + str(l.get_loss(c).sum()))
            nn = input()
            if nn != '':
                break
    m,n = 0,0
    with open('model.pkl','wb') as f:
        pickle.dump(l,f)
    for i in range(500):
        b = a[0][i].reshape((28 * 28, 1))
        c = a[1][i].reshape((10, 1))
        ans = l.predict(b).reshape(10).tolist()
        ans = ans.index(max(ans))
        c = c.reshape(10).tolist().index(1)
        if ans == c:
            m += 1
        else:
            n += 1
        print(ans, '\t', c)
    print('\n%d----%d'%(m,n))
    import os
    os.system('pause')



def test():
    n,m = 0,0
    with open('model.pkl','rb') as f:
        l = pickle.load(f)
    a = get_data()
    for i in range(25000, 30000):
        b = a[0][i].reshape((28 * 28, 1))
        c = a[1][i].reshape((10, 1))
        ans = l.predict(b).reshape(10).tolist()
        ans = ans.index(max(ans))
        c = c.reshape(10).tolist().index(1)
        if ans == c:
            m += 1
        else:
            n += 1
        print(ans, '\t', c)
    print('\n%d----%d'%(m,n))
    import os
    os.system('pause')

main()
