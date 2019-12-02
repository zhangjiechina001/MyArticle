import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend
backend.set_image_data_format('channels_first')


def loadData(path="mnist.npz"):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

# 从Keras导入Mnist数据集
(x_train, y_train), (x_validation, y_validation) = loadData()

# 设定随机种子
seed = 7
np.random.seed(seed)

x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_validation = x_validation.reshape(x_validation.shape[0], 1, 28, 28).astype('float32')
# 格式化数据到0~1
x_train = x_train/255
x_validation = x_validation/255

# 进行one-hot编码
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)

# 定义模型
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


model = create_model()
model.fit(x_train, y_train, epochs=2, batch_size=50, verbose=2)

score = model.evaluate(x_validation, y_validation, verbose=0)
print('CNN_Small: %.2f%%' % (score[1]*100))
filename='pretectNumModel.h5'
model.save(filename)
print('save success!')
