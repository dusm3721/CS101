import tensorflow as tf
import numpy as np

###     Creating Graph and Running it in a Session      ###

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x * x * y + y + 2

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()

with tf.Session() as sess:
    sess.run(x.initializer)
    sess.run(y.initializer)
    result = sess.run(f)

    
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()    
    
sess = tf.InteractiveSession()
init.run()
result = f.eval()
sess.close()    



###     Linear Regression with TensorFlow       ###

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
N, p = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((N,1)), housing.data]

X = tf.constant(housing_data_plus_bias)
y = tf.constant(housing.target.reshape(N,1))
beta = tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(X), X)), tf.matmul(tf.transpose(X), y)) 

with tf.Session() as sess:
    beta_value = beta.eval()
    
uhat = housing.target.reshape(N,1) - np.matmul(housing_data_plus_bias, beta_value)    
RMSE = np.sum(uhat ** 2) / uhat.size
SSR = np.sum(uhat ** 2)
SST = np.sum((housing.target - np.mean(housing.target)) ** 2)
Rsquared = 1 - SSR / SST


###     Autodiff        ###

x = tf.Variable(2.)
y = tf.Variable(3.)
f = x ** 2 + y ** 2
fprime = tf.gradients(f, [x,y])

with tf.Session() as session:
    x.initializer.run()
    y.initializer.run()
    f_val = session.run(f)
    fprime_val = session.run(fprime)
    
print(f_val)
print(fprime_val)    


###     Linear Regression from Batch Gradient Descent       ###
n_epoch = 1000
gamma = 0.01

# Standardizing input data is important
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(housing_data_plus_bias[:, 1:])
housing_data_standardized = scaler.transform(housing_data_plus_bias[:, 1:])

housing_data_standardized_plus_bias = np.c_[np.ones([N,1]), housing_data_standardized]

X = tf.constant(housing_data_standardized_plus_bias, dtype=tf.float32)
y = tf.constant(housing.target.reshape(N,1), dtype=tf.float32)
beta = tf.Variable(tf.random_uniform([p+1,1],-1.0,1.0), dtype=tf.float32)
yhat = tf.matmul(X, beta)
uhat = y - yhat
mse = tf.reduce_mean(tf.square(uhat))

gradients = tf.gradients(mse, beta)
training_op = tf.assign(beta, beta - tf.multiply(gamma, gradients)[0])

# Directly using Optimizer #
optimizer = tf.train.GradientDescentOptimizer(learning_rate=gamma)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epoch):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE=", sess.run(mse))
        sess.run(training_op)
    best_beta = sess.run(beta)





###     Linear Regression from Online Gradient Descent       ###
n_epoch = 5
gamma = 0.01
    
X = tf.placeholder(tf.float32, shape=(None, p+1))
y = tf.placeholder(tf.float32, shape=(None, 1))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epoch):
        print("Epoch", epoch, "MSE=", sess.run(mse))
        for i in range(N):
            X_point, y_point = (housing_data_standardized_plus_bias[i].reshape(1,p+1), housing.target[i].reshape(1,1))
            sess.run(training_op, feed_dict={X:X_point, y:y_point})
    best_beta = beta.eval()
                
    
