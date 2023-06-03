import tensorflow.compat.v1 as tf
import numpy as np

tf.compat.v1.enable_eager_execution()

(train_images,train_labels),(test_images,test_labels)= tf.keras.datasets.mnist.load_data()

#step 2 (Forward propagaion)
model=tf.keras.Sequential()
#model.add는 layer을 추가함
model.add(tf.keras.layers.Dense(512,activation=tf.nn.relu,input_shape=(784,)))
#First dense layer(512 units), dense = 각각의 x가 모든 z에 연결되어있다. 인풋의 개수는 784 = 28 * 28, 히든 유닛은 512, relu함수가 액티베이션 함수이디.

model.add(tf.keras.layers.Dense(256,activation=tf.nn.relu))
#Second dense layers(256 units)

model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
#Output (10 units) == y가 1일 확률과 0일 확률이 된다.

#Finish step 2 (backword propagation)
model.compile(loss='categorical_crossentropy',optimizer='rmsprop')
#loss == sigma ~~ , rmsprop이라는 optimizer를 사용한다.
#optimizer = adam, sgd, adadelta 등등이 존재한다. 경우에 따라 다름

model.fit(train_images,train_labels,epochs=5)
#fw pro 와 bw pro를 5번 해라 ( 트레이닝 시키기 )

#step 4
loss,accuracy = model.evaluate(test_images,test_labels)
print("Accuracy", accuracy)
#step 5
scores = model.predict(test_images[0:1]) #1번째 이미지를 활용하여 점수를 predict
print(np.argmax(scores))  #가장 높은 evidence값을 가지는 outputlayer를 출력

# import tensorflow.compat.v1 as tf

# tf.compat.v1.enable_eager_execution() #for loop 사용 가능
# for image,label in dataset:           #session없이 현재 상황을 확인 가능함.
#     print(train_images)
