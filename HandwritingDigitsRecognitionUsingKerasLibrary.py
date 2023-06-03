import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
print(tf.__version__)

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train,axis=1)
y_trian=

print(x_train[0])

plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()

model=tf.keras.models.Sequential()    #layer 1이 layer 2가 연결되어 있다.Sequencial하게
model.add(tf.keras.layers.Flatten())  #2차원 데이터를 1차원 데이터로 플랫하게 만듬
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
#first layer
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
#second layer
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
#output layer

model.compile(optimizer= 'adam',loss='sparse+categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)

val_loss,val_acc = model.evaluate(x_test,y_test)
print("val_loss_function=",val_loss)
print("val_accuracy=",val_acc)

predictions = model.predict(x_test)
print(predictions)    #교수님도 잘 모름, 암튼 prediction임
print(np.argmax(predictions[0]))

plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()

#디버깅할 때는 assert를 사용하기도 한다.(Debugging Aid)
#koras를 안 쓰고 tf만으로도 딥러닝 구현이 가능하긴 하다.(Lecture 23 p.55, 이 이후로는 생략하심)
