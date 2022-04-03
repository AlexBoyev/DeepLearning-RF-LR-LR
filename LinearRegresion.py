from __future__ import print_function
import tensorflow as tf
import numpy
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

rng= numpy.random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

learning_rate= 0.01
training_epochs= 1000
display_step= 50
data = pd.read_excel('coffee-data.xlsx') #read from dataset
#mapping
data['Taste Preference'] = data['Taste Preference'].map({'black': 1, 'milk/cream': 2})
data['Tempature Preference'] = data['Tempature Preference'].map({'frozen': 3, 'iced': 4, "hot": 5})

input_X = data[["Cups of coffee in one week?", "Cups of coffee outside home per week?","Taste Preference"]]
input_Y = data["Tempature Preference"]
input_X = numpy.asmatrix(input_X)
input_Y = numpy.array(input_Y).reshape((input_Y.shape[0],1))

train_X, test_X, train_Y, test_Y = train_test_split(input_X, input_Y, test_size=0.33)

#train_X= float(train_X)
#train_Y= float(train_Y)
#train_X= numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
#7.042,10.791,5.313,7.997,5.654,9.27,3.1])
#train_Y= numpy.asarray([1.7,1.7,1.7,1.7,1.7,1.7,1.7,1.7,1.7,1.7,
#1.7,1.7,1.7,1.7,1.7,1.7,1.7])

n_samples= train_X.shape[0]

X = tf.placeholder("float")
Y = tf.placeholder("float")
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

pred= tf.add(tf.multiply(X, W), b)
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init= tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (input_X, input_Y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: input_X, Y: input_Y})

    if (epoch+1) % display_step== 0:
        c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
            "W=", sess.run(W), "b=", sess.run(b))
    print("Optimization Finished!")



    training_cost= sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X+ sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    # Testing example, as requested (Issue #2)
    #test_X = numpy.asarray([3, 1, 4, 3, 9, 3, 1, 5, 8])
    #test_Y = numpy.asarray([8, 5, 1, 3, 9, 3, 4, 1, 3])
    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
    tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
    feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))
    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
