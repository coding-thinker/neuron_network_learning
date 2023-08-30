import gzip
import os 
import pickle
import numpy as np
from model import MLP
import mlflow


def load_mnist():
    def one_hotify(y):
        return np.eye(10)[y].reshape(10, 1)
    # Load data
    with gzip.open(os.path.join(os.curdir, "data", "mnist.pkl.gz"), "rb") as data_file:
        train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    target_shape = (784, 1)

    train_inputs = [x.reshape(target_shape) for x in train_data[0]]
    train_results = [one_hotify(y) for y in train_data[1]]
    train_data = list(zip(train_inputs, train_results))

    val_inputs = [x.reshape(target_shape) for x in val_data[0]]
    val_results = val_data[1]
    val_data = list(zip(val_inputs, val_results))

    test_inputs = [x.reshape(target_shape) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))
    return train_data, val_data, test_data





if __name__ == "__main__":
    np.random.seed(42)

    num_neurons = [784, 256, 64, 10]
    lr = 0.01
    batch_size = 16
    epochs = 100
    activation = 'relu'
    mlflow.log_params({
        "num_neurons": num_neurons,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "activation": activation
    })

    # Initialize train, val and test data
    train_set, dev_set, test_set = load_mnist()

    # dataset pre analysis
    # print(f"{len(train_set)=}, {len(dev_set)=}, {len(test_set)=}")
    # import matplotlib.pyplot as plt
    # import collections

    # cnt = collections.Counter([y.argmax() for _, y in train_set])
    # b = plt.bar(cnt.keys(), cnt.values())
    # plt.bar_label(b)
    # plt.xticks(range(10), range(10))
    # plt.title("Distribution of label on the train set")
    # plt.show()

    # plt.suptitle("Sample images from the MNIST")
    # for i in range(150, 158):
    #     plt.subplot(2, 4, i-149)
    #     plt.imshow(train_set[i][0].reshape(28, 28), cmap="gray")
    
    # plt.show()


    # n = np.zeros_like(train_set[0][0])
    # for x, _ in train_set:
    #     n += x / len(train_set)
    # plt.imshow(n.reshape(28, 28), cmap="gray")
    # plt.show()



    mlp = MLP(
        nodes_per_layer=num_neurons, 
        lr=lr, 
        bs=batch_size, 
        activation=activation,  
    )
    mlp.train(train_set, dev_set, epochs)

    test_acc = mlp.calc_acc(test_set) / len(test_set) * 100
    print(f"{test_acc=}%.")
    mlflow.log_metric("test_acc", test_acc)
