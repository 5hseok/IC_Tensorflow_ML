import tensorflow.compat.v1 as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
xy=np.array([[1,2,1,0],[1,3,2,0],[1,3,5,0],[1,5,5,1],[1,7,5,1],[1,2,5,1]],dtype='float32')  #데이터들

x_train = xy[0:4,0:3]           #0~3까지의 데이터로 훈련시키고
y_train = xy[0:4,3:4]           #x는 0~2,y는 3번 인덱스 값

x_test = xy[4:6,0:3]            #4~5까지의 데이터는 훈련이 잘 되었나 확인하는 용도
y_test = xy[4:6,3:4]

print(x_train,x_train.shape)
print(y_train,y_train.shape)
print(x_test)
print(y_test)

#Parameters
learning_rate = 0.01
training_epochs = 25
num_features = x_train.shape[1]

#tf Graph input
X=tf.placeholder(tf.float32, [None,num_features])
Y=tf.placeholder(tf.float32,[None,1])

#Set model weights
W=tf.Variable(tf.zeros([num_features,1]))
b=tf.Variable(tf.zeros([1]))
Z=tf.add(tf.matmul(X,W),b)
prediction = tf.nn.sigmoid(Z)

#Calculate the cost
cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z,labels=Y))

#Use Adam as optimization method

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
cost_history = np.empty(shape=[1],dtype=float)

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        _, c =sess.run([optimizer,cost], feed_dict={X:x_train,Y:y_train})
        print("Epoch: ", '%04d' % (epoch+1), "\ncost=", "{: .9f} ".format(c), "\nW=\n",sess.run(W),"\nb=",sess.run(b))
        cost_history = np.append(cost_history,c)
        
#Calculate the correct predictions
    correct_prediction = tf.cast(tf.greater(prediction,0/5),dtype=tf.float32)

#Calculrate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Y,correct_prediction),dtype=tf.float32))         #tf.cast(parameter,dtype)을 bool에서 float로 바꿀 때 사용

    print("Train Accuracy: ",accuracy.eval({X: x_train,Y: y_train}))
    print("Test Accuracy: ",accuracy.eval({X:x_test,Y:y_test}))

sess.close()
