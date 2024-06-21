#import libraries and packages
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

#load the data
data = mnist.load_data()

#split the data into train and test
(x_train, y_train), (x_test, y_test) = data

#reshape the data and change its format
x_train = x_train.reshape((x_train.shape[0], 28*28)).astype('float32')
x_test = x_test.reshape((x_test.shape[0], 28*28)).astype('float32')

#normalize the data
x_train = x_train/255
x_test = x_test/255

#create the model
model = Sequential()
model.add(Dense(32, input_shape=(28,28), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

#compile the model
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

#display model summary
model.summary()

#train the model
model.fit(x_train, y_train, epochs=10, batch_size=100)

#test the model
scores = model.evaluate(x_test, y_test)
print("Accuracy: ", scores[1]*100)
