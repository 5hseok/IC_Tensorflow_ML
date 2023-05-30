import tensorflow as tf
import numpy as np

b=tf.Variable(tf.zeros((100,)))                  #b는 초기값을 100개의 0으로 설정
W=tf.Variable(tf.random_uniform((784,100),-1,1)) #w와 b는 Variable이다. 
#uniform은 -1부터 1까지 숫자를 뽑는다고 했을 때 모두 똑같은 확률을 가지게 한다. 
#즉 784 X 100개의 -1부터 1까지의 값을 초기 세팅값으로 설정한다.
x=tf.placeholder(tf.float32,(100,784))
#x는 placeholder
h=tf.nn.relu(tf.matmupusl(x,W)+b)
#x와 W를 곱하고 b를 더한 후 relu function을 취하여 h를 구한다.
#이런 개수의 연산은 크기가 매우 크기 때문에 CPU보단 GPU가 더 효율적이다. 
#이 코드를 실행하기 위해 Session을 지정해야 한다.
sess=tf.Session()
sess.run(tf.initialize_all_variables())     #session에 초기값을 설정한다.
sess.run(h,{x: np.random.random(100,784)})  #h라는 sess을 실행해라. 실행에 필요한 x는 100 x 784개의 무작위 데이터를 준다.
#이때 betch = 100

# Use placeholder for labels
# Build loss node using labels and prediction 
prediction=h
label=tf.placeholder(tf.float32,[100,10]) #Y와 같다
cross_entropy = -tf.reduce_sum(label * tf.log(prediction),axis=1)
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#0.5 = 알파 , cross_entropy = J
sess=tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(1000):
    batch_x,batch_y = data.next_batch()
    sess.run(train_step,feed_dict={x: batch_x,label: batch_y})