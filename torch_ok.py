# see: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

from __future__ import print_function, division
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import nn
import numpy as np
from sklearn import metrics


df = pd.read_csv("./data/imdb_archive/contentDataPrime.csv")
df = df.loc[df['releaseYear'] > 0]

X = df.releaseYear
y = df.rating

X_train = X[:70000]
y_train = y[:70000]
X_test = X[70000:]
y_test = X[70000:]

# преобразуем данные в тензоры PyTorch
X_train = torch.tensor(X_train.values, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(y_train.values, dtype=torch.long)
X_test = torch.tensor(X_test.values, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.long)

model = nn.Linear(1, 10)
criterion = torch.nn.MultiMarginLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(X_train)
    # вычисляем потери на тренировочных данных
    loss = criterion(y_pred, y_train)
    # обнуляем градиенты оптимизатора
    optimizer.zero_grad()
    # вычисляем градиенты потерь по параметрам модели
    loss.backward()
    # обновляем параметры модели с помощью оптимизатора
    optimizer.step()
    # выводим значение потерь каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

with torch.no_grad():  # we don't need gradients in the testing phase
    y_pred = model(X_test)
    y_pred = torch.argmax(y_pred, dim=1)

    print("Accuracy score:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))

# plt.clf()
# plt.plot(X_train, y_train, 'go', label='True data', alpha=0.5)
# plt.plot(X_train, predicted, '--', label='Predictions', alpha=0.5)
# plt.legend(loc='best')
# plt.show()

