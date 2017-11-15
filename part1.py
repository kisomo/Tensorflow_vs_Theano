#https://medium.com/@sentimentron/faceoff-theano-vs-tensorflow-e25648c31800

import tensorflow as tf
import numpy as np

# Make 100 phony data points in NumPy.
x_data = np.float32(np.random.rand(2, 100)) # Random input
y_data = np.dot([0.100, 0.200], x_data) + 0.300

print(x_data.shape)
print(x_data[:,0:4])
print(y_data.shape)
print(y_data[0:4])

# Construct a linear model.
b = tf.Variable(tf.zeros([1]))
print(b)
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
print(W)
y = tf.matmul(W, x_data) + b
print(y.shape)

# Minimize the squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# For initializing the variables.
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Fit the plane.
for step in xrange(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(b)

# Learns best fit is W: [[0.100  0.200]], b: [0.300]

print("================================================================")

import theano
import theano.tensor as T
import numpy

# Again, make 100 points in numpy
x_data = numpy.float32(numpy.random.rand(2, 100))
y_data = numpy.dot([0.100, 0.200], x_data) + 0.3
print(x_data.shape)
print(x_data[:,0:5])
print(y_data.shape)
print(y_data[0:5])

# Intialise the Theano model
X = T.matrix()
print(X.shape)
Y = T.vector()
print(Y.shape)
b = theano.shared(numpy.random.uniform(-1, 1), name="b")
print(b.shape)
W = theano.shared(numpy.random.uniform(-1.0, 1.0, (1, 2)), name="W")
print(W.shape)
y = W.dot(X) + b 
print(y.shape)

# Compute the gradients WRT the mean-squared-error for each parameter
cost = T.mean(T.sqr(y - Y))
gradientW = T.grad(cost=cost, wrt=W)
gradientB = T.grad(cost=cost, wrt=b)
updates = [[W, W - gradientW * 0.5], [b, b - gradientB * 0.5]] 
print(updates)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True) 
print(train)


for i in xrange(0, 201):
    train(x_data, y_data)
    print W.get_value(), b.get_value()


