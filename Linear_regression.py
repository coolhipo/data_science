import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from math import sqrt
from sklearn.model_selection import cross_val_score
from mpl_toolkits.mplot3d import Axes3D

#importing dataset
dataset = pd.read_csv("Development.csv")

#x,y designation
x = np.array(dataset.drop(columns= ["Area (sq. mi.)", "Pop. Density ", "Development Index", "Infant mortality "]))
y = np.array(dataset["Infant mortality "])

#plotting graphs
plot1 = dataset.plot(x= "GDP ($ per capita)", y='Infant mortality ', style='o')
plot2 = dataset.plot(x= "Literacy (%)", y='Infant mortality ', style='o')
plot3 = dataset.plot(x= "Population", y='Infant mortality ', style='o')
plt.tight_layout()
plt.show()

#shapiro stats
stat, p = shapiro(y)
print(f"показатель {p}")
print(f"статистика {stat}")

#data split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print(f"regressor.intercept_ {regressor.intercept_}")
print(f"regressor.coef_{regressor.coef_}")

#cross validation
scores = cross_val_score(regressor, x, y, cv=5)
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

#prediction
y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

#comparision of predictions and real values
df1 = df.head(30)
df1.plot(kind = 'bar')
plt.grid(which ='major', color='black')
plt.grid(which='minor', color='red')
plt.show()

#RMSE in %
print(sqrt(mean_squared_error(y_test, y_pred)))

#regression plot
plt.scatter(x_test[0:,1], y_test)
plt.plot(x_test[0:,1],  (regressor.intercept_ + x_test[0:,1] * regressor.coef_[1]), color='orange', linewidth=1 )
plt.show()
