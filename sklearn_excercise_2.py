import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import metrics

btc_price = pd.read_csv('btc.csv')
# print(btc_price.head())

X = btc_price.loc[0:499, ["Close Price"]]
y = btc_price.loc[1:500, ["Close Price"]]

# print(X.head)
# print(y.head)

X_train = np.array(X[:400])
y_train = np.array(y[:400])
X_test = np.array(X[400:])
y_test = np.array(y[400:])

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# print("Accuracy score:", metrics.accuracy_score(y_test, y_pred))

plt.plot(range(0, 100), y_test, label='Actual price', color='blue')
plt.plot(range(0, 100), y_pred, label='Predicted price', color='red')
plt.xlabel('Hours')
plt.ylabel('Price ($)')
plt.grid(True)
plt.legend(loc="upper left")

plt.show()
