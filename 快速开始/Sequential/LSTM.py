from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import Embedding
from keras.layers import LSTM

import numpy as np
import keras
# 生成虚拟数据

x_train = np.random.random((1000, 16))
y_train = np.random.random((1000, 1))

# 生成虚拟验证数据
x_test = np.random.random((100,  16))
y_test = np.random.random((100, 1))

max_features = 1024

model = Sequential()
model.add(Embedding(max_features,output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

model.fit(x_train,
          y_train,
          epochs=20,
          batch_size=16)
score = model.evaluate(x_test,y_test,batch_size=16)
print('loss:',score[0])
print('accuracy:',score[1])