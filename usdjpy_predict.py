import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import metrics

usdjpy_price = pd.read_csv("usdjpy.csv")

X = usdjpy_price.loc[0:260, ["Close"]]
y = usdjpy_price.loc[1:261, ["Close"]]

X_train = np.array(X[:205])
y_train = np.array(y[:205])
X_test = np.array(X[205:])
y_test = np.array(y[205:])

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# print("Predicated:\n", y_pred)
# print("Test:\n", y_test)

# print(y_pred.shape)
# print(y_test.shape)

plt.plot(range(0, 56), y_test, label='Actual price', color='blue')
plt.plot(range(0, 56), y_pred, label='Predicted price', color='red')
plt.xlabel('Days')
plt.ylabel('Price')
plt.grid(True)
plt.legend(loc="upper left")

plt.show()
