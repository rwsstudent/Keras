import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import SGD
import numpy as np

#生成虚拟数据
x_train = np.random.random((100,100,100,3))
y_train = keras.utils.to_categorical(np.random.randint(10,size=(100,1)),num_classes=10)
x_test = np.random.random((20,100,100,3))
y_test = keras.utils.to_categorical(np.random.randint(10,size=(20,1)),num_classes=10)

model = Sequential()
#输入：3通道100x100像素
#使用32个大小为3x3的卷积滤波器
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)
model.fit(x_train,
          y_train,
          epochs=20,
          batch_size=32)
score = model.evaluate(x_test,y_test,batch_size=32)

print('loss:',score[0])
print('accuracy:',score[1])
