from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPool2D
from keras import backend
import matplotlib.pyplot as plot
from keras.callbacks import TensorBoard

batch_size=128
num_class=10
epochs=10

img_rows,img_cols=28,28

(x_train,y_train),(x_test,y_test)=mnist.load_data(r'C:\Users\Administrator\PycharmProjects\MyPython\pyQt5\keras\mnist.npz')
import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(x_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()
if backend.image_data_format()=='channels_first':
    x_train=x_train.reshape(x_train.shape[0],1,img_rows,img_cols)
    x_test = x_train.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape=(1,img_rows,img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0],  img_rows, img_cols,1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols,1)
    input_shape = ( img_rows, img_cols,1)

x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train=x_train/255
x_test=x_test/255
print('x_trainshape',x_train.shape)
print('train samples:',x_train.shape[0])
print('test samples',x_test.shape[0])


y_train=keras.utils.to_categorical(y_train,num_class)
y_test=keras.utils.to_categorical(y_test,num_class)
xtemp=x_train[0]
model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test),
          callbacks=[TensorBoard(log_dir='mytensorboard/3')])

score=model.evaluate(x_test,y_test,verbose=0)
model.save('CNNModel.h5',overwrite=True,include_optimizer=True)
print('Test loss:',score[0])
print('Test accuracy:',score[1])

