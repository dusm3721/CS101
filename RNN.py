import tensorflow as tf
import numpy as np

n_inputs = 3
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
b = tf.Variable(tf.zeros(shape=[1, n_neurons], dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(X1, Wx) + tf.matmul(Y0, Wy) + b)

init = tf.global_variables_initializer()

X0_batch = np.array([[0,1,2], [3,4,5], [6,7,8], [9,0,1]])
X1_batch = np.array([[9,8,7], [0,0,0], [6,5,4], [3,2,1]])

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0,Y1], feed_dict={X0 : X0_batch, X1 : X1_batch})

print(Y0_val)
print(Y1_val)

###     Create RNN cell from tf built-in functions      ###
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, final_state = tf.contrib.rnn.static_rnn(basic_cell, [X0,X1], dtype=tf.float32)

Y0, Y1 = output_seqs

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0,Y1], feed_dict={X0 : X0_batch, X1 : X1_batch})

print(Y0_val)
print(Y1_val)


###     Dynamic Unrolling RNN through time      ###
n_inputs = 3
n_neurons = 5
n_steps = 2

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])   # (n, t, p)
seq_length = tf.placeholder(tf.int32, [None])   # the lengths of input sequences (input data can have different lenghts of seq) 

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)       # Create an RNN cell (default activation=tanh)
output_seqs, final_state = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=seq_length)    # Create an RNN network

Y = output_seqs

init = tf.global_variables_initializer()

X_batch = np.array([
        # step0     #step1
        [[0,1,2], [9,8,7]], # instance 0
        [[3,4,5], [0,0,0]], # instance 1 (seq length=1, padded with zeros)
        [[6,7,8], [6,5,4]], # instance 2
        [[9,0,1], [3,2,1]], # instance 3
        ])
seq_length_batch = np.array([2,1,2,2])

with tf.Session() as sess:
    init.run()
    Y_val = sess.run(Y, feed_dict={X : X_batch, seq_length : seq_length_batch})

print(Y_val)


###     RNN for MNIST classification    ###
n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10

learning_rate = 0.01

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs_seqs, final_state = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = tf.layers.dense(final_state, n_outputs)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

n_epochs = 100
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
        acc_test = accuracy.eval(feed_dict={X:X_test, y:y_test})
        print(epoch, "Training accuracy:", acc_train, "test accuracy:", acc_test)
        
        
###     Predict Time Series     ###
t = np.arange(0, 30, 0.1)
ts_data = t * np.sin(t) / 3 + 2 * np.sin(5*t)

import matplotlib
import matplotlib.pyplot as plt
plt.plot(t, ts_data)


n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.tanh)
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_fc_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
y_hat = tf.reshape(stacked_fc_outputs, [-1, n_steps, n_outputs])

learning_rate = 0.0005

loss = tf.reduce_mean(tf.square(y[:,-1,:] - y_hat[:,-1,:]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

X_test = np.array([ts_data[i:i+20].reshape(-1,1) for i in range(250, 265)])
y_test = np.array([ts_data[i+1:i+1+20].reshape(-1,1) for i in range(250,265)])

saver = tf.train.Saver()

n_epochs = 30

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for i in np.random.permutation(200):
            X_batch, y_batch = ts_data[i : i+20].reshape(1,-1,1), ts_data[i+1 : i+1+20].reshape(1,-1,1)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        mse = sess.run(loss, feed_dict={X:X_test, y:y_test})
        print("epoch:", epoch, "mse=", mse)
        saver.save(sess, "./my_model_final.ckpt")

ts_gen = list(ts_data[100:120])

with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    for i in range(250):
        X_batch = np.array(ts_gen[-n_steps:]).reshape(1, n_steps, 1)
        y_hat_val = sess.run(y_hat, feed_dict={X:X_batch})
        ts_gen.append(y_hat_val[0,-1,0])

plt.plot(t[100:300], ts_gen[0:200], t[0:300], ts_data[0:300])


###     Predict Time Series Deep RNN       ###
t = np.arange(0, 30, 0.1)
ts_data = t * np.sin(t) / 3 + 2 * np.sin(5*t)

import matplotlib
import matplotlib.pyplot as plt
plt.plot(t, ts_data)


n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

cell1 = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.tanh)
cell2 = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.tanh)
multi_layer_cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2])
rnn_outputs, final_state = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_fc_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
y_hat = tf.reshape(stacked_fc_outputs, [-1, n_steps, n_outputs])

learning_rate = 0.005

loss = tf.reduce_mean(tf.square(y[:,-1,:] - y_hat[:,-1,:]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

X_test = np.array([ts_data[i:i+20].reshape(-1,1) for i in range(250, 265)])
y_test = np.array([ts_data[i+1:i+1+20].reshape(-1,1) for i in range(250,265)])

saver = tf.train.Saver()

n_epochs = 10

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for i in np.random.permutation(200):
            X_batch, y_batch = ts_data[i : i+20].reshape(1,-1,1), ts_data[i+1 : i+1+20].reshape(1,-1,1)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        mse = sess.run(loss, feed_dict={X:X_test, y:y_test})
        print("epoch:", epoch, "mse=", mse)
        saver.save(sess, "./my_model_final.ckpt")

ts_gen = list(ts_data[100:120])

with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    for i in range(250):
        X_batch = np.array(ts_gen[-n_steps:]).reshape(1, n_steps, 1)
        y_hat_val = sess.run(y_hat, feed_dict={X:X_batch})
        ts_gen.append(y_hat_val[0,-1,0])

plt.plot(t[100:300], ts_gen[0:200], t[0:300], ts_data[0:300])

     
###     Predict Time Series Deep RNN       ###
t = np.arange(0, 30, 0.1)
ts_data = t * np.sin(t) / 3 + 2 * np.sin(5*t)

import matplotlib
import matplotlib.pyplot as plt
plt.plot(t, ts_data)


n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

cell1 = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.tanh)
cell2 = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.tanh)
multi_layer_cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2])
rnn_outputs, final_state = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

y_hat = tf.layers.dense(rnn_outputs[:,-1,:], n_outputs)

learning_rate = 0.005


loss = tf.reduce_mean(tf.square(y[:,-1,:] - y_hat[:,:]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

X_test = np.array([ts_data[i:i+20].reshape(-1,1) for i in range(250, 265)])
y_test = np.array([ts_data[i+1:i+1+20].reshape(-1,1) for i in range(250,265)])

saver = tf.train.Saver()

n_epochs = 10

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for i in np.random.permutation(200):
            X_batch, y_batch = ts_data[i : i+20].reshape(1,-1,1), ts_data[i+1 : i+1+20].reshape(1,-1,1)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        mse = sess.run(loss, feed_dict={X:X_test, y:y_test})
        print("epoch:", epoch, "mse=", mse)
        saver.save(sess, "./my_model_final.ckpt")

ts_gen = list(ts_data[100:120])

with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    for i in range(250):
        X_batch = np.array(ts_gen[-n_steps:]).reshape(1, n_steps, 1)
        y_hat_val = sess.run(y_hat, feed_dict={X:X_batch})
        ts_gen.append(y_hat_val[0,0])

plt.plot(t[100:300], ts_gen[0:200], t[0:300], ts_data[0:300])
        
        
        
        
        
        
        