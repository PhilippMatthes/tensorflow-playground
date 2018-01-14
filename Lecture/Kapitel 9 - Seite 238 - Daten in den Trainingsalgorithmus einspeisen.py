import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
import numpy.random as rnd


housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

A = tf.placeholder(tf.float32, shape=(None, 3))
print(A)
B = A + 5
print(B)
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8 ,9]]})

print(B_val_1)
print(B_val_2)


X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

batch_size = 100
n_batches = int(np.ceil(m / batch_size))
n_epochs = 1000

learning_rate = 0.01

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2 / m * tf.matmul(tf.transpose(X), error)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

saver = tf.train.Saver()

def fetch_batch(epoch, batch_index, batch_size):
    rnd.seed(epoch * n_batches + batch_index)
    indices = rnd.randint(m, size=batch_size)
    return housing_data_plus_bias[indices], housing.target.reshape(-1, 1)[indices]

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")