import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras import Sequential
import numpy as np

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
# print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)
# 对数据归一化
x_train = x_train / 255
x_test = x_test / 255


model = tf.keras.Sequential()
# 输入层，将28*28的二维图像展成一维
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# 隐藏层1
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
# 隐藏层2
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
# 隐藏层3
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
# 输出层
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# 损失函数
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

# 模型训练，迭代5次
model.fit(x_train, y_train, epochs=5)
result = model.evaluate(x_test, y_test) 
print('TEST ACC:', result[1])