import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import tensorflow as tf

#seeds
numpy.random.seed(42)

#picture size
img_rows, img_cols= 28, 28

#data loading and splitting
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#creating image size
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

#data normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# model building and teaching
model2 = Sequential()

model2.add(Conv2D(75, kernel_size=(5, 5),activation='sigmoid', input_shape=input_shape))
model2.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

model2.add(Conv2D(100, (5, 5), activation='sigmoid'))
model2.add(MaxPooling2D(pool_size=(1, 1)))
# model.add(Dropout(0.2))

model2.add(Conv2D(100, (5, 5), activation='sigmoid'))
model2.add(MaxPooling2D(pool_size=(1, 1)))
# model.add(Dropout(0.2))
model2.add(Flatten())

model2.add(Dense(500, activation='sigmoid'))
# model.add(Dropout(0.5))
model2.add(Dense(10, activation='softmax'))

opt = tf.keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.95,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    name="RMSprop",)


model2.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model21 = model2.fit(x_train, y_train, epochs=3, validation_split=0.2)

# accuracy on test data
scores = model2.evaluate(x_test, y_test, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))

#plot of teaching speed
plt.plot(model21.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
plt.plot(model21.history['val_accuracy'], 
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

#testing on your own images
image = cv2.imread(r'C:\Users\coolh\Desktop\Study\machine learning\lab4\test3.png', cv2.IMREAD_GRAYSCALE)
image1 = cv2.resize(image, (28, 28))
image1 = image1.astype('float32')
image1 = image1.reshape(1, 28, 28, 1)
image1 = 255-image1
image1 /= 255
image

pred = model2.predict(image1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(numpy.argmax(pred, axis=1))
plt.show()
