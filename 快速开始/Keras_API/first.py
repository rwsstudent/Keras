
'''
开始使用Keras函数式API
'''
#Keras函数式API是定义复杂模型的方法，比如：多输出模型、有向无环图，或具有共享层的模型

'''
例一：全连接网络
Sequential模型可以很好的实现全连接网络，但这个例子可以更好的理解
网络层的实例是可以调用的，他以张量为参数，并返回一个张量
输入和输出均为张量，它们都可以用来定义叶哥模型Model
'''

from keras.layes import Input,Dense
from keras.models import Model

#这部分返回一个张量
inputs = Input(shape = (784,))
#层的实例是可调用的，它以张量为参数，并返回一个张量
x = Dense(64,activation ='relu')(inputs)
x = Dense(63,activation = 'relu')(x)
predictions = Dense(10,activation = 'softmax')(x)
#这部分创建了一个包含输入层和三个全连接层的模型
model = Model(inputs = inputs,outputs = predictions)
model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy'.
              metricts = ['accuracy'])
model.fit(data,lables)#开始训练

'''
所有的模型都可以调用，就像网络层一样
利用函数式API，可以轻易的重用训练好的模型；可以将任何模型看作一个层，然后通过传递一个张量来调用它。
调用模型时，不仅重用了模型也重用了权重
'''
x = Input(shape= (784,))
#这是可行的，并且返回上面定义的10-way softmax
y = model(x)

'''
这种方式允许我们快速创建可以处理序列输入的模型，只需一行代码，就可以将图像分类模型转换为视频分类模型
'''
from keras.layers import TimeDistributed
#输入张量时20个时间步的序列
#每个时间为一个784维的向量
input_sequences = Input(shape = (20,784))
#这部分将之前定义的模型应用到输入序列中的每个时间步
#之前定义的模型的输出时一个10-way softmax
#因此下面层的输出是维度为10的20个向量序列
processed_sequences = TimDistributed(model)(input_sequences)



          
