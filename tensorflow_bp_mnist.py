import tensorflow.compat.v1 as tf
import numpy as np
import pickle
import os
tf.disable_v2_behavior()



def inserting(n):
    temp = [0 for i in range(10)]
    temp[n] = 1
    return temp


def get_data(table=0):
    with open('mnist.pkl', 'rb') as f:
        data_set = pickle.load(f, encoding='iso-8859-1')
    data_train_list = [data_set[table][0][i].reshape((28*28, 1)).copy() for i in range(data_set[table][0].shape[0])]
    label_train_list = [np.array(inserting(j)).reshape((10, 1)) for j in data_set[table][1]]
    return data_train_list, label_train_list


def train_and_test():

    BATCH_SIZE = 1

    x_all, y_all = get_data()
    x_data = x_all[:]
    y_data = y_all[:]

    x = tf.placeholder(tf.float32, shape=(28 * 28, 1))
    y = tf.placeholder(tf.float32, shape=(10, 1))
    w0 = tf.Variable(tf.random.uniform([16, 28 * 28], -1.0, 1.0))
    b0 = tf.Variable(tf.zeros([16, 1]))

    w1 = tf.Variable(tf.random.uniform([10, 16], -1.0, 1.0))
    b1 = tf.Variable(tf.zeros([10, 1]))

    z1 = tf.add(tf.matmul(w0, x), b0)
    o1 = tf.nn.sigmoid(z1)
    z2 = tf.add(tf.matmul(w1, o1), b1)
    o2 = tf.nn.sigmoid(z2)

    loss = tf.reduce_mean(tf.square(tf.subtract(o2, y)))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    saver =  tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        STEPS = 45000
        for i in range(STEPS):
            for _ in range(50):
                sess.run(train_step, feed_dict={x: x_data[i], y: y_data[i]})
            if i % 500 == 0:
                # 每训练500个steps打印训练误差
                total_loss = sess.run(loss, feed_dict={x: x_data[i], y: y_data[i]})
                print("After %d training step(s), loss_mse on all data is %g" % (i, total_loss))
        m,n = 0,0
        for i in range(45000, 50000):
            ans = sess.run(o2, feed_dict={x: x_data[i]})
            ans = ans.reshape(10).tolist()
            ans = ans.index(max(ans))
            c = y_data[i].reshape(10).tolist().index(1)
            print(ans, '-----', c)
            if ans == c:
                m += 1
            else:
                n += 1
        print('Accuracy percentage:', m/(m+n)*100)
        save_path = saver.save(sess,"tensorflow_bp_mnist/tensorflow_bp_mnist_model.ckpt")
        print('saved to:', save_path)
    os.system('pause')


def test():
    x_all, y_all = get_data(1)
    x_data = x_all[:]
    y_data = y_all[:]

    x = tf.placeholder(tf.float32, shape=(28 * 28, 1))
    y = tf.placeholder(tf.float32, shape=(10, 1))
    w0 = tf.Variable(tf.random.uniform([16, 28 * 28], -1.0, 1.0))
    b0 = tf.Variable(tf.zeros([16, 1]))

    w1 = tf.Variable(tf.random.uniform([10, 16], -1.0, 1.0))
    b1 = tf.Variable(tf.zeros([10, 1]))

    z1 = tf.add(tf.matmul(w0, x), b0)
    o1 = tf.nn.sigmoid(z1)
    z2 = tf.add(tf.matmul(w1, o1), b1)
    o2 = tf.nn.sigmoid(z2)

    loss = tf.reduce_mean(tf.square(tf.subtract(o2, y)))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    
    saver = tf.train.Saver()
    m,n = 0,0
    with tf.Session() as sess:
        saver.restore(sess, "tensor_bp_model/tensor_bp.ckpt")
        for i in range(len(x_data)):
            ans = sess.run(o2, feed_dict={x: x_data[i]})
            ans = ans.reshape(10).tolist()
            ans = ans.index(max(ans))
            c = y_data[i].reshape(10).tolist().index(1)
            print(ans, '-----', c)
            if ans == c:
                m += 1
            else:
                n += 1
        print('Accuracy percentage:', m/(m+n)*100)
    os.system('pause')

train_and_test()    
