"""
Этот код работает, но я его совершенно не понимаю
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from sklearn.preprocessing import normalize # импортируем функцию нормализации из sklearn

df = pd.read_csv("./data/imdb_archive/contentDataPrime.csv")
df = df.loc[df['releaseYear'] > 0]

X = df.releaseYear
y = df.rating

X_train = X[:70000]
y_train = y[:70000]
X_test = X[70000:]
y_test = y[70000:]

# # применяем нормализацию к данным, используя функцию normalize из sklearn
# # нормализуем данные по столбцам (axis=0) с нормой L2 (по умолчанию)
# X_train_norm = normalize(X_train.values.reshape(1, -1), axis=0).reshape(-1, 1)
# y_train_norm = normalize(y_train.values.reshape(1, -1), axis=0).reshape(-1, 1)
# X_test_norm = normalize(X_test.values.reshape(1, -1), axis=0).reshape(-1, 1)
# y_test_norm = normalize(y_test.values.reshape(1, -1), axis=0).reshape(-1, 1)

# преобразуем данные в тензоры PyTorch
X_train = torch.tensor(X_train.values, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test.values, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

# создаем модель линейной регрессии с помощью полносвязного слоя
model = nn.Linear(1, 1) # линейный слой с одним входом и одним выходом
criterion = nn.MSELoss() # функция потерь среднеквадратичная ошибка (MSE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # оптимизатор SGD

# обучаем модель
epochs = 100 # количество эпох обучения
for epoch in range(epochs):
    # вычисляем предсказания модели на тренировочных данных
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
    # if (epoch + 1) % 10 == 0:
    #     print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# проверяем модель на тестовых данных
with torch.no_grad(): # отключаем вычисление градиентов для ускорения
    # вычисляем предсказания модели на тестовых данных
    y_pred = model(X_test)
    # вычисляем MSE на тестовых данных
    mse = criterion(y_pred, y_test)
    # выводим MSE и RMSE на тестовых данных
    print(f"MSE: {mse.item():.4f}")
    print(f"RMSE: {np.sqrt(mse.item()):.4f}")

    # добавляем построение графика с помощью matplotlib на модель
    # рисуем точки тестовых данных в синем цвете
    plt.scatter(X_test.numpy(), y_test.numpy(), c='b', label='data')
    # рисуем линию предсказаний модели в красном цвете
    plt.plot(X_test.numpy(), y_pred.numpy(), c='r', label='prediction')

    print(y_pred)

    # добавляем подписи осей и легенду
    plt.xlabel('releaseYear')
    plt.ylabel('rating')
    plt.legend()
    # показываем график
    # plt.show()
