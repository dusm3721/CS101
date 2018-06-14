#############################
#       1 layer RNN         #
#############################

import tensorflow as tf
import numpy as np

n_steps = 20
n_inputs = 1
n_neurons1 = 100
n_neurons2 = 50
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

cell1 = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons1, activation=tf.nn.tanh)

multi_layer_cell = tf.contrib.rnn.MultiRNNCell([cell1])
rnn_outputs, last_state = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
y_hat = tf.layers.dense(rnn_outputs[:,-1,:], n_outputs)

learning_rate = 0.005

loss = tf.reduce_mean(tf.square(y - y_hat))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


t = np.arange(0, 30, 0.1)
ts_data = t * np.sin(t) / 3 + 2 * np.sin(5*t)
import matplotlib
import matplotlib.pyplot as plt
plt.plot(t, ts_data)

X_test = np.array([ts_data[i:i+20].reshape(-1,1) for i in range(250, 265)])
y_test = np.array([ts_data[i+20].reshape(1) for i in range(250,265)])

n_epochs = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for i in np.random.permutation(200):
            X_batch, y_batch = ts_data[i : i+20].reshape(1,-1,1), ts_data[i+20].reshape(1,1)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        mse = sess.run(loss, feed_dict={X:X_test, y:y_test})
        print("epoch:", epoch, "mse=", mse)
    saver.save(sess, "./my_model_final.ckpt")

ts_gen = list(ts_data[0:20])

with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    for i in range(280):
        X_batch = np.array(ts_gen[-n_steps:]).reshape(1, n_steps, 1)
        y_hat_val = sess.run(y_hat, feed_dict={X:X_batch})
        ts_gen.append(y_hat_val[0,0])

plt.plot(t[0:300], ts_gen[0:300], t[0:300], ts_data[0:300])
        
#############################
#       2 layers RNN        #
#############################
import tensorflow as tf
import numpy as np

n_steps = 20
n_inputs = 1
n_neurons1 = 100
n_neurons2 = 50
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

cell1 = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons1, activation=tf.nn.tanh)
cell2 = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons2, activation=tf.nn.tanh)
multi_layer_cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2])
rnn_outputs, last_state = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
y_hat = tf.layers.dense(rnn_outputs[:,-1,:], n_outputs)

learning_rate = 0.005

loss = tf.reduce_mean(tf.square(y - y_hat))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


t = np.arange(0, 30, 0.1)
ts_data = t * np.sin(t) / 3 + 2 * np.sin(5*t)
import matplotlib
import matplotlib.pyplot as plt
plt.plot(t, ts_data)

X_test = np.array([ts_data[i:i+20].reshape(-1,1) for i in range(250, 265)])
y_test = np.array([ts_data[i+20].reshape(1) for i in range(250,265)])

n_epochs = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for i in np.random.permutation(200):
            X_batch, y_batch = ts_data[i : i+20].reshape(1,-1,1), ts_data[i+20].reshape(1,1)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        mse = sess.run(loss, feed_dict={X:X_test, y:y_test})
        print("epoch:", epoch, "mse=", mse)
    saver.save(sess, "./python/rnn_2layers.ckpt")

ts_gen = list(ts_data[0:20])

with tf.Session() as sess:
    saver.restore(sess, "./python/rnn_2layers.ckpt")
    for i in range(280):
        X_batch = np.array(ts_gen[-n_steps:]).reshape(1, n_steps, 1)
        y_hat_val = sess.run(y_hat, feed_dict={X:X_batch})
        ts_gen.append(y_hat_val[0,0])

plt.plot(t[0:300], ts_gen[0:300], t[0:300], ts_data[0:300])
        



#############################
#       1 layer LSTM        #
#############################

import tensorflow as tf
import numpy as np

n_steps = 20
n_inputs = 1
n_neurons1 = 100
n_neurons2 = 50
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons1, activation=tf.nn.tanh)

multi_layer_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell])
rnn_outputs, last_state = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
y_hat = tf.layers.dense(rnn_outputs[:,-1,:], n_outputs)

learning_rate = 0.005

loss = tf.reduce_mean(tf.square(y - y_hat))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


t = np.arange(0, 30, 0.1)
ts_data = t * np.sin(t) / 3 + 2 * np.sin(5*t)
import matplotlib
import matplotlib.pyplot as plt
plt.plot(t, ts_data)

X_test = np.array([ts_data[i:i+20].reshape(-1,1) for i in range(250, 265)])
y_test = np.array([ts_data[i+20].reshape(1) for i in range(250,265)])

n_epochs = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for i in np.random.permutation(200):
            X_batch, y_batch = ts_data[i : i+20].reshape(1,-1,1), ts_data[i+20].reshape(1,1)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        mse = sess.run(loss, feed_dict={X:X_test, y:y_test})
        print("epoch:", epoch, "mse=", mse)
    saver.save(sess, "./my_model_final.ckpt")

ts_gen = list(ts_data[0:20])

with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    for i in range(280):
        X_batch = np.array(ts_gen[-n_steps:]).reshape(1, n_steps, 1)
        y_hat_val = sess.run(y_hat, feed_dict={X:X_batch})
        ts_gen.append(y_hat_val[0,0])

plt.plot(t[0:300], ts_gen[0:300], t[0:300], ts_data[0:300])



#############################
#       2 layers LSTM       #
#############################

import tensorflow as tf
import numpy as np

n_steps = 20
n_inputs = 1
n_neurons1 = 100
n_neurons2 = 50
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons1, activation=tf.nn.tanh)
lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons2, activation=tf.nn.tanh)

multi_layer_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell1, lstm_cell2])
rnn_outputs, last_state = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
y_hat = tf.layers.dense(rnn_outputs[:,-1,:], n_outputs)

learning_rate = 0.005

loss = tf.reduce_mean(tf.square(y - y_hat))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


t = np.arange(0, 30, 0.1)
ts_data = t * np.sin(t) / 3 + 2 * np.sin(5*t)
import matplotlib
import matplotlib.pyplot as plt
plt.plot(t, ts_data)

X_test = np.array([ts_data[i:i+20].reshape(-1,1) for i in range(250, 265)])
y_test = np.array([ts_data[i+20].reshape(1) for i in range(250,265)])

n_epochs = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for i in np.random.permutation(200):
            X_batch, y_batch = ts_data[i : i+20].reshape(1,-1,1), ts_data[i+20].reshape(1,1)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        mse = sess.run(loss, feed_dict={X:X_test, y:y_test})
        print("epoch:", epoch, "mse=", mse)
    saver.save(sess, "./python/lstm_2layers.ckpt")

ts_gen = list(ts_data[0:20])

with tf.Session() as sess:
    saver.restore(sess, "./python/lstm_2layers.ckpt")
    for i in range(280):
        X_batch = np.array(ts_gen[-n_steps:]).reshape(1, n_steps, 1)
        y_hat_val = sess.run(y_hat, feed_dict={X:X_batch})
        ts_gen.append(y_hat_val[0,0])

plt.plot(t[0:300], ts_gen[0:300], t[0:300], ts_data[0:300])


        
        
        
        
        
        