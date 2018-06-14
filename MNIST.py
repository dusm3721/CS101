import numpy as np

###     Load and pre-process MNIST data     ###
 
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original")
X = mnist["data"]
y = mnist["target"]
N, p = X.shape
shuffle_index = np.random.permutation(N)
X_train , y_train = X[shuffle_index[0:60000]], y[shuffle_index[0:60000]]
X_test, y_test = X[shuffle_index[60000:N]], y[shuffle_index[60000:N]]

'''
from sklearn.preprocessing import StandardScaler  # Standardization MNIST data performs terribly
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
'''

from sklearn.preprocessing import MinMaxScaler   # Scale the input in range [0,1]
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

def accuracy(y, yhat):
    count = 0
    for i in range(len(y)):
        if y[i] == yhat[i]: count += 1
    return count / len(y)


###     Logistic Regression Classification      ###

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class="multinomial", solver="sag", max_iter=10)
model.fit(X_train, y_train)

yhat_train = model.predict(X_train)
yhat_test = model.predict(X_test)

acc_train = accuracy(y_train, yhat_train)
acc_test = accuracy(y_test, yhat_test)

model.predict_proba(X_test[0:5])
yhat_test[0:5]
y_test[0:5]


###     Random Forest Classification        ###

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

yhat_train = model.predict(X_train)
yhat_test = model.predict(X_test)

acc_train = accuracy(y_train, yhat_train)
acc_test = accuracy(y_test, yhat_test)

model.predict_proba(X_test[0:5])
yhat_test[0:5]
y_test[0:5]


###############################################
###         DNN from TensorFlow             ###
###############################################

import tensorflow as tf

n_inputs = p
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10 

# Create placeholders for inputs and targets
X = tf.placeholder(tf.float32, shape=(None, n_inputs))
y = tf.placeholder(tf.int64, shape=(None))

# Build neuron layers and create DNN
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)
logits = tf.layers.dense(hidden2, n_outputs)    # no specification of activation means Linear

# Define the cost function
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)     # logit is the output before entering the Softmax function
loss = tf.reduce_mean(cross_entropy)

# Create an optimizer
gamma = 0.005
optimizer = tf.train.GradientDescentOptimizer(learning_rate = gamma)
training_op = optimizer.minimize(loss)

# Define a performance measure
def accuracy(y, yhat):
    count = 0
    for i in range(len(y)):
        if y[i] == yhat[i]: count += 1
    return count / len(y)

# Create an initializer
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Excution phase
n_epoch = 25
batch_size = 1

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoch):
        for batch in range(len(X_train) // batch_size):
            X_batch, y_batch = X_train[batch*batch_size : (batch+1)*batch_size], y_train[batch*batch_size : (batch+1)*batch_size]
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        ylogits_train, ylogits_test = sess.run(logits, feed_dict={X:X_train}), sess.run(logits, feed_dict={X:X_test})    
        yhat_train, yhat_test = np.argmax(ylogits_train, axis=1), np.argmax(ylogits_test, axis=1)
        acc_train, acc_test = accuracy(y_train, yhat_train), accuracy(y_test, yhat_test)
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    saver.save(sess, "./my_model_final.ckpt")

# Using the trained DNN
with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    ylogits_test = sess.run(logits, feed_dict={X:X_test})
    yprob_test = sess.run(tf.nn.softmax(ylogits_test))

yhat_test = np.argmax(yprob_test, axis=1)
yprob_test[0:5]
yhat_test[0:5]
y_test[0:5]


################################################################

'''
import tensorflow as tf

n_inputs = p
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10 

X = tf.placeholder(tf.float32, shape=(None, n_inputs))
y = tf.placeholder(tf.int64, shape=(None))

hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)
logits = tf.layers.dense(hidden2, n_outputs)    # no specification of activation means Linear

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)     # logit is the output before entering Softmax function
loss = tf.reduce_mean(cross_entropy)

gamma = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate = gamma)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

n_epoch = 40
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoch):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
        acc_test = accuracy.eval(feed_dict={X:mnist.test.images, y:mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    save_path = saver.save(sess, "./my_model_final.ckpt")

with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    Z = logits.eval(feed_dict={X:X_test})
    yhat = np.argmax(Z, axis=1)

print(yhat)
print(y_test)
'''


###############################################
###         RNN from TensorFlow             ###
###############################################

import tensorflow as tf

n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10

# Treat each row as a state of input sequence 
X_train = X_train.reshape(-1,28,28)
X_test = X_test.reshape(-1,28,28)

# Create placeholders for inputs and targets
X = tf.placeholder(tf.float32, shape=(None, n_steps, n_inputs))
y = tf.placeholder(tf.int64, shape=(None))

# Build RNN network
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs_seqs, final_state = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
logits = tf.layers.dense(final_state, n_outputs)

# Define the cost function
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)     # logit is the output before entering the Softmax function
loss = tf.reduce_mean(cross_entropy)

# Create an optimizer
gamma = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate = gamma)
training_op = optimizer.minimize(loss)

# Define a performance measure
def accuracy(y, yhat):
    count = 0
    for i in range(len(y)):
        if y[i] == yhat[i]: count += 1
    return count / len(y)

# Create an initializer
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Excution phase
n_epoch = 25
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoch):
        for batch in range(len(X_train) // batch_size):
            X_batch, y_batch = X_train[batch*batch_size : (batch+1)*batch_size], y_train[batch*batch_size : (batch+1)*batch_size]
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        ylogits_train, ylogits_test = sess.run(logits, feed_dict={X:X_train}), sess.run(logits, feed_dict={X:X_test})    
        yhat_train, yhat_test = np.argmax(ylogits_train, axis=1), np.argmax(ylogits_test, axis=1)
        acc_train, acc_test = accuracy(y_train, yhat_train), accuracy(y_test, yhat_test)
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    saver.save(sess, "./my_model_final.ckpt")

# Using the trained DNN
with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    ylogits_test = sess.run(logits, feed_dict={X:X_test})
    yprob_test = sess.run(tf.nn.softmax(ylogits_test))

yhat_test = np.argmax(yprob_test, axis=1)
yprob_test[0:5]
yhat_test[0:5]
y_test[0:5]

