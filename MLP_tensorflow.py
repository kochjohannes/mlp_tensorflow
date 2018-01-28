import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate data

phi_tmp = np.arange(0, 6*np.pi, 6*np.pi/50)
x_tmp = np.arange(0, 1, 1/50)
c0 = np.array([[np.multiply(x_tmp, np.cos(phi_tmp))], [np.multiply(x_tmp, np.sin(phi_tmp))]])
c1 = np.array([[0.8*np.multiply(x_tmp, np.cos(phi_tmp))], [0.8*np.multiply(x_tmp, np.sin(phi_tmp))]])
c = np.concatenate([c0, c1], axis=2)
c = c.reshape([2, 100])
X_data = c.transpose()
Y_data = np.zeros([100, 1])
Y_data[0:50] = Y_data[0:50] + 1
plt.plot(X_data[0:50,0], X_data[0:50,1], 'ro')
plt.plot(X_data[50:100,0], X_data[50:100,1], 'bx')
plt.show()

learning_rate = 0.001
training_epochs = 100000
batch_size = 10

# Define network / Create computational graph
n_hidden_1 = 100 #units in layer 1
n_hidden_2 = 100 #units in layer 2
n_hidden_3 = 100 #units in layer 2
n_input = 2 #input dimensions
n_classes = 1 #output dimension

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def multilayer_perceptron(X):
    #Input to layer (X or output from previous layer, is multiplied with the
    #weight matrix, and then the bias is added, finally the output of the layer
    #is the activation function of XW+b, here the sigmoid)
    layer_1 = tf.sigmoid(tf.add(tf.matmul(X, weights['h1']), biases['b1']))
    layer_2 = tf.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    layer_3 = tf.sigmoid(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    #Note that I didn't include the sigmoid in the output layer, I will
    #add this later in the code
    out_layer = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
    return out_layer

logits = multilayer_perceptron(X)

loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

#Initializing the variables
init = tf.global_variables_initializer()

#with tf.Session() as sess:
sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    #avg_cost = 0.
    #total_batch = int(X.shape[0]/batch_size)
    loss = sess.run([train_op, loss_op], feed_dict={X: X_data,
                                                   Y: Y_data})

    #print("Epoch:", epoch, "t Cost: ", loss[1])
print("Training done")

pred = tf.nn.sigmoid(logits)
correct_prediction = tf.equal(tf.argmax(Y_data,1), tf.argmax(pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Accuracy: ", sess.run(accuracy, feed_dict={X: X_data, Y:Y_data}))
prediction = tf.argmax(pred, 1)
boundary_x = []
boundary_y = []
for x_grid in np.linspace(-1, 1, 300):
    for y_grid in np.linspace(-1,1,300):
        prob = pred.eval(feed_dict={
            X: np.reshape(np.array([x_grid, y_grid]), [1,2])},
                         session=sess)
        if(prob>=0.495 and prob<=0.505):
            boundary_x.append(x_grid)
            boundary_y.append(y_grid)

plt.plot(X_data[0:50,0], X_data[0:50,1], 'ro')
plt.plot(X_data[50:100,0], X_data[50:100,1], 'bx')
plt.plot(boundary_x, boundary_y, 'c.')
plt.show()
