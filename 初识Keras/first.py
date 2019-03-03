
'''
Keras 是一个用 Python 编写的高级神经网络 API，它能够以 TensorFlow, CNTK, 或者 Theano 作为后端运行.
'''

'''
Keras 的核心数据结构是model,一种组织网络层的方式。
最简单的模型是Sequential模型，有多个网络层线性堆叠。
复杂模型Keras函数式API，可以构建任意的神经网络图。
'''

'''
Sequential模型如下：
'''
from keras.models import Sequential
model = Sequential()

'''
也可以简单使用.add()来堆叠模型:
'''
from keras.layers import Dense
model.add(Dense(units = 64,activation = 'relu',input_dim = 100))
model.add(Dense(units = 10,activation = 'softmax'))

'''
在完成模型的构建后，可以使用.compile()来配置学习过程
'''
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'sgd',
              metrics = ['accuracy'])

'''
也可以进一步配置优化器
'''
model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.SGD(lr = 0.01,momentum  = 0.9,nesterov = True))

'''
训练迭代数据
x_train,y_train是Numpy数组
'''
model.fit(x_train,y_train,epochs = 5,batch_size = 32)

'''
模型评估
'''
loss_and_metrics = model.evalute(x_test,y_test,batch_size = 128)

'''
对新数据进行生成预测
'''
classes = model.predict(x_test,batch_size = 128)

              



                
                

           
