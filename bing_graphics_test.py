# Импортируем библиотеки
import numpy as np
import matplotlib.pyplot as plt

# Создаем массив x от 0 до 10 с шагом 0.1
x = np.arange(0, 10, 0.1)

# Создаем массивы y1 и y2 как функции от x
y1 = np.sin(x)
y2 = np.cos(x)

# Рисуем графики y1 и y2 по x с разными цветами и метками
plt.plot(x, y1, color='red', label='sin(x)')
plt.plot(x, y2, color='blue', label='cos(x)')

# Добавляем заголовок, подписи осей и легенду
plt.title('Графики синуса и косинуса')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Показываем график на экране
plt.show()