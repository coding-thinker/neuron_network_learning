import numpy as np
from matplotlib import pyplot as plt


def set_sample(a_in, b_in, xrange_in, fluctuation_in, size_in):
    x = np.random.normal(xrange_in[0], xrange_in[1], size=size_in)
    return x, x * a_in + b_in + np.random.normal(0, fluctuation_in, size=size_in)


def linear_func(k_in, b_in, x_in):
    # k*x + b
    return k_in * x_in + b_in


def loss_func(y_in, y_expected_in):
    # 1/2n * sigma((y-y0)^2)
    n = y_in.size
    return ((y_in - y_expected_in) ** 2).sum() / (2 * n)


def gradient_calculator_k(data_x, data_y, func, k_in, b_in):
    m = data_x.size
    expectation_y = func(k_in, b_in, data_x)
    return ((expectation_y - data_y) * data_x).sum() / m


def gradient_calculator_b(data_x, data_y, func, k_in, b_in):
    m = data_x.size
    expectation_y = func(k_in, b_in, data_x)
    return (expectation_y - data_y).sum() / m


def gradient_decenter(data_x, data_y, climb, bias, rate, trip):
    if data_x.size == data_y.size:
        for _ in range(trip):
            loss = loss_func(linear_func(climb, bias, data_x), data_y)
            delta_climb = gradient_calculator_k(data_x, data_y, linear_func, climb, bias) * rate
            delta_bias = gradient_calculator_b(data_x, data_y, linear_func, climb, bias) * rate
            climb -= delta_climb
            bias -= delta_bias
            if loss < 10:
                pass
        print((climb, bias))
        return (climb, bias)
    else:
        return 1  # raise data size error


if __name__ == '__main__':
    xrange = (0, 100)
    a = 5.23143
    b = -3.24425
    rate = 0.0001
    data_x, data_y = set_sample(a, b, xrange, 0.4, (100))
    climb = 0
    bias = 0
    trip = 100000
    climb, bias = gradient_decenter(data_x, data_y, climb, bias, rate, trip)
    plt.scatter(data_x, data_y, s=1)
    plt.plot([data_x.min(), data_x.max()], [linear_func(climb, bias, data_x.min()), linear_func(climb, bias, data_x.max())])
    plt.show()
