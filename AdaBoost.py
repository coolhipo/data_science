import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import multilabel_confusion_matrix

#importing data
data = pd.read_csv("Development.csv")

#plotting graphs
plot1 = data.plot(x= "GDP ($ per capita)", y='Development Index', style='o')
plot2 = data.plot(x= "Literacy (%)", y='Development Index', style='o')
plot3 = data.plot(x= "Infant mortality ", y='Development Index', style='o')
plt.tight_layout()
plt.show()

#data normalization
scaler = MinMaxScaler()
norm_array = scaler.fit_transform(data)
norm_df = pd.DataFrame(norm_array, columns= data.columns)

#x,y designation
x = pd.DataFrame(norm_df, columns=['GDP ($ per capita)', 'Literacy (%)', 'Infant mortality '])
y = pd.DataFrame(data, columns=['Development Index'])

#data split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)

#creating and traning the model
AdaModel = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
model = AdaModel.fit(x_train, y_train.values.ravel())
y_pred = model.predict(x_test)

#accuracy
print('Точность:', metrics.accuracy_score(y_test, y_pred))

#comparision of predictions and real values
df = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
df1 = df.head(30)
df1.plot(kind = 'bar')
plt.grid(which ='major', color='black')
plt.grid(which='minor', color='red')
plt.show()

#f1 scores
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
print(f'f1_micro = {f1_macro}, f1_micro = {f1_micro}, f1_weighted = {f1_weighted}')

#confusion matrix
multilabel_confusion_matrix(y_test, y_pred)
