import os
import numpy as np
import random
import functional as f
from alive_progress import alive_bar
import scipy
import math
from typing import *
import mlflow

class MLP:
    """An implement for MLP
    """

    def __init__(
        self,
        nodes_per_layer=[784, 30, 10],
        lr=1e-2,
        bs=16,
        activation="relu"
    ):

        # Hyperparameters setting
        self.nodes_per_layer = nodes_per_layer
        self.num_layers = len(nodes_per_layer)
        self.activation = getattr(f, activation)
        self.activation_derivative = getattr(f, f"{activation}_derivative")
        self.lr = lr
        self.bs = bs
        # weight for layers
        self.W = [np.zeros((1,))] + [np.random.randn(y, x)/np.sqrt(x) for y, x in zip(nodes_per_layer[1:], nodes_per_layer[:-1])]

        # bias for layers
        self.b = [np.zeros((1,))] + [np.random.randn(y, 1) for y in nodes_per_layer[1:]]

        # z = w@x + b
        self.z = [np.zeros_like(b) for b in self.b]

        # a = activation(z)
        self.a = [np.zeros_like(b) for b in self.b]

 

    def train(self, 
              train_set:List[Tuple[np.ndarray]], 
              dev_set:List[Tuple[np.ndarray]]=None, 
              epochs: int=100
        ):
        # shuffle the set
        random.shuffle(train_set)
        # split into batches
        batches = scipy.array_split(train_set, math.ceil(len(train_set) / self.bs))
        with alive_bar(epochs) as bar:
            for epoch in range(epochs):
                for batch in batches:
                    # per batch training and gradient descent per batch
                    self.train_batch(batch)
                train_acc = self.calc_acc_onehot(train_set) / len(train_set) * 100
                print(f"{epoch=}, {train_acc=} %.")
                mlflow.log_metric("train_acc", train_acc, step=epoch)
                

                if dev_set is not None:
                    val_acc = self.calc_acc(dev_set) / len(dev_set) * 100
                    print(f"{epoch=}, {val_acc=} %.")
                    mlflow.log_metric("val_acc", val_acc, step=epoch)

                bar()

    def train_batch(self, batch):
        gradient_b = [np.zeros(bias.shape) for bias in self.b]
        gradient_W = [np.zeros(weight.shape) for weight in self.W]
        for x, y in batch:
            self.forward_propaganda(x)
            delta_b, delta_W = self.back_propaganda(y)
            for i in range(len(gradient_b)):
                gradient_b[i] += delta_b[i]

            for i in range(len(gradient_W)):
                gradient_W[i] += delta_W[i]


        # gradient descent for all weights and biases
        for i in range(len(self.W)):
            self.W[i] -= (self.lr / self.bs) * gradient_W[i]

        for i in range(len(self.b)):
            self.b[i] -= (self.lr / self.bs) * gradient_b[i]

    def forward_propaganda(self, x):
        self.a[0] = x
        # per layer forward
        for i in range(1, self.num_layers):
            self.z[i] = (
                self.W[i] @ self.a[i - 1] + self.b[i]
            )
            if i == self.num_layers - 1:
                self.a[i] = f.softmax(self.z[i])
            else:
                self.a[i] = self.activation(self.z[i])

    def back_propaganda(self, y):
        delta_b = [np.zeros_like(b) for b in self.b]
        delta_W = [np.zeros_like(w) for w in self.W]

        gradient = (self.a[-1] - y)
        delta_b[-1] = gradient
        delta_W[-1] = gradient @ self.a[-2].transpose()
        # per layer backward
        for l in range(self.num_layers - 2, 0, -1):
            gradient = self.W[l + 1].transpose() @ gradient
            gradient = gradient * self.activation_derivative(self.z[l])
        
            delta_b[l] = gradient
            delta_W[l] = gradient @ self.a[l - 1].transpose()

        return delta_b, delta_W

    def calc_acc(self, dataset):
        correct = [(self.infer(x) == y) for x, y in dataset]
        return sum(result for result in correct)
    
    def calc_acc_onehot(self, dataset):
        correct = [(self.infer(x) == y.argmax()) for x, y in dataset]
        return sum(result for result in correct)

    def infer(self, x):
        self.forward_propaganda(x)
        return np.argmax(self.a[-1])
        