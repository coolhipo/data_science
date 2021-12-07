import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import multilabel_confusion_matrix
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

# load data
dataset = pd.read_csv('Development.csv')
dataset.sample(frac=1)
y = np.array(dataset.pop('Development Index'))

#plot graphs
plot1 = data.plot(x= "GDP ($ per capita)", y='Development Index', style='o')
plot2 = data.plot(x= "Literacy (%)", y='Development Index', style='o')
plot3 = data.plot(x= "Infant mortality ", y='Development Index', style='o')
plt.tight_layout()
plt.show()

# data split
x = pd.DataFrame(dataset, columns=['GDP ($ per capita)', 'Literacy (%)', 'Infant mortality '])
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.33, random_state=0)
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
# data normalization
scalerX = MinMaxScaler(feature_range=(0, 1))
scalerY = MinMaxScaler(feature_range=(0, 1))
scalerX = scalerX.fit(x.to_numpy())
scalerY = scalerY.fit(y.reshape(len(y), 1))

x_train_scale = scalerX.transform(x_train)
x_test_scale = scalerX.transform(x_test)
y_train_scale = scalerY.transform(y_train.reshape(len(y_train), 1))

scalerX = MinMaxScaler()
scalerY = MinMaxScaler()
x1 = scalerX.fit_transform(x)
y1 = scalerY.fit_transform(y.reshape(-1, 1))

#building and training neural network
model = Sequential()
model.add(Dense(9, activation='sigmoid'))
model.add(Dense(9, activation='sigmoid'))
model.add(Dense(9, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
opt = tf.keras.optimizers.RMSprop(
    learning_rate=0.1,
    rho=0.95,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    name="RMSprop",)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])
model1 = model.fit(x_train_scale, y_train_scale , epochs=200, batch_size = 10)

#confusion matrix
from sklearn.metrics import multilabel_confusion_matrix
preds = model.predict(x_test_scale)
preds1 = scalerY.inverse_transform(preds)
preds2 = np.rint(preds1)

#comparision of predictions and real values
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': preds1.flatten().round()})
df1 = df.head(30)
df1.plot(kind = 'bar')
plt.grid(which ='major', color='black')
plt.grid(which='minor', color='red')
plt.show()
multilabel_confusion_matrix(y_test, preds2)
