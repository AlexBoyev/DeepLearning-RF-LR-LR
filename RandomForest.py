# Random Forest Classification on Tensorflow


# Import
from __future__ import print_function
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
from sklearn.model_selection import train_test_split
import csv
import xlrd
import unicodecsv
import os
import win32com.client

#mapping
data_xls = pd.read_excel('coffee-survey.xlsx')
data_xls['Student'] = data_xls['Student'].map({'no': 1, 'yes': 2})
data_xls['Drink Coffee'] = data_xls['Drink Coffee'].map({ 'No': 1,'Yes': 2,})
data_xls['Job'] = data_xls['Job'].map({'None': 1, 'No, Half Time': 2, "Yes, Full Time": 3})
#converting to csv and deleting columns

with open("coffee-survey1.csv", "w") as my_csv:
	pass
my_csv.close()
data_xls.to_csv('coffee-survey1.csv', encoding='utf-8', index=False)
f=pd.read_csv("coffee-survey1.csv")
keep_col = ['Job','Sleep Hours','Number of 12 oz cups']
new_f = f[keep_col]
new_f.to_csv("coffee-survey2.csv", index=False)
data = pd.read_csv('coffee-survey2.csv')
os.remove("coffee-survey1.csv")

data = pd.read_csv('coffee-survey2.csv')
# Extract feature and target np arrays (inputs for placeholders)
input_x = data.iloc[:, 0:-1].values
input_y = data.iloc[:, -1].values
#split the data 33% and 66%
X_train, X_test, y_train, y_test = train_test_split(input_x, input_y, test_size=0.33, random_state=0) # split

# Parameters
num_steps = 250 # Total steps to train
num_classes = 11
num_features = 2
num_trees = 10
max_nodes = 1000

# Input and Target placeholders
X = tf.placeholder(tf.float32, shape=[None, num_features])
Y = tf.placeholder(tf.int32, shape=[None])
# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes, num_features=num_features, num_trees=num_trees,max_nodes=max_nodes).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)

# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)
# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))
# Start TensorFlow session
sess = tf.Session()
# Run the initializer
sess.run(init_vars)
# Training
for i in range(1, num_steps + 1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    # batch_x, batch_y = mnist.train.next_batch(batch_size)
    _, l = sess.run([train_op, loss_op], feed_dict={X: X_train, Y: y_train})
    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X: X_train, Y: y_train})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))
# Test Model
#test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: X_test, Y: y_test}))
os.remove("coffee-survey2.csv")