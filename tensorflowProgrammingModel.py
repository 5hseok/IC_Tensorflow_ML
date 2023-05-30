import tensorflow as tf

b=tf.Variable(tf.zeros((100,)))
W=tf.Variable(tf.random_uniform((784,100),-1,1))
x=tf.placeholder(tf.float32,(100,784))
h=tf.nn.relu(tf.matmul(x,W)+b)