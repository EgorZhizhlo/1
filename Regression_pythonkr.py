import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
# %%
# Импортируем необходимую функцию
from sklearn.datasets import fetch_openml

# Загрузка датасета "wine-quality-red" из OpenML
dataset = fetch_openml(name='wine-quality-red', version=1, as_frame=True)

# Выводим текстовое описание датасета
print(dataset.DESCR)

# Обозначаем целевую переменную за y, а остальные данные за X
X = dataset.data
y = dataset.target

# Дополнительно можно вывести первые несколько строк для проверки
print(X.head())
print(y.head())

# %%
num_rows, num_columns = X.shape
print(f"Число строк (объектов): {num_rows}")
print(f"Число столбцов (признаков): {num_columns}\n")

# 2. Информация о датасете
print("Информация о датасете:")
print(X.info())
print("\n")

# 3. Статистическое описание признаков
print("Статистическое описание признаков:")
print(X.describe())
print("\n")

# 4. Дополнительно: Проверка на пропущенные значения
print("Количество пропущенных значений в каждом признаке:")
print(X.isnull().sum())
# %%
# Проверка на пропущенные значения в X
missing_values_X = X.isnull().sum()
total_missing_X = missing_values_X.sum()

# Проверка на пропущенные значения в y
missing_values_y = y.isnull().sum()
total_missing_y = missing_values_y.sum()

print("Количество пропущенных значений в признаках (X):")
print(missing_values_X)
print(f"Общее количество пропущенных значений в X: {total_missing_X}\n")

print("Количество пропущенных значений в целевой переменной (y):")
print(missing_values_y)
print(f"Общее количество пропущенных значений в y: {total_missing_y}\n")

# Заполнение пропущенных значений медианными значениями (если имеются)
if total_missing_X > 0:
    print("Заполнение пропущенных значений в X медианными значениями.")
    X = X.fillna(X.median())
else:
    print("В данных X нет пропущенных значений.\n")

if total_missing_y > 0:
    print("Заполнение пропущенных значений в y медианными значениями.")
    y = y.fillna(y.median())
else:
    print("В целевой переменной y нет пропущенных значений.\n")

# %%
sns.set(style="whitegrid")
# Построение гистограммы с KDE
plt.figure(figsize=(10, 6))
sns.histplot(y, bins=30, kde=True, color='skyblue')
plt.title('Распределение целевой переменной')
plt.xlabel('Значения целевой переменной')
plt.ylabel('Частота')
plt.show()

# %%
import numpy as np
import pandas as pd

# Предполагаем, что X и y уже определены и очищены из предыдущих шагов
# Преобразуем X и y в numpy массивы
X = X.values  # Преобразование DataFrame в numpy массив
y = y.values.astype(float)  # Преобразование Series в numpy массив с типом float

# Добавим столбец единиц для смещения (bias term) в матрицу признаков
X = np.hstack((X, np.ones((X.shape[0], 1))))  # Последний столбец - bias

# Обновим список признаков для ясности
features = list(dataset.data.columns) + ['bias']

# Просмотр формы данных
print(f"Форма X: {X.shape}")
print(f"Форма y: {y.shape}")

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Проверка и преобразование X в numpy массив, если необходимо
if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
    X = X.values
    print("X преобразован из pandas DataFrame/Series в numpy.ndarray.")
else:
    print("X уже является numpy.ndarray.")

if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
    y = y.values.astype(float)
    print("y преобразован из pandas DataFrame/Series в numpy.ndarray.")
else:
    print("y уже является numpy.ndarray.")

# Проверка на NaN или бесконечные значения
print("\nПроверка X на NaN и бесконечные значения:")
print("NaN:", np.isnan(X).sum())
print("Inf:", np.isinf(X).sum())

print("\nПроверка y на NaN и бесконечные значения:")
print("NaN:", np.isnan(y).sum())
print("Inf:", np.isinf(y).sum())

# Масштабирование признаков
scaler = StandardScaler()
X_features = X[:, :-1]  # Все столбцы кроме последнего (bias)
X_scaled = scaler.fit_transform(X_features)
X = np.hstack((X_scaled, X[:, -1].reshape(-1, 1)))  # Последний столбец - bias

print("\nФорма X после масштабирования:", X.shape)

# Инициализация весов случайными небольшими значениями
n_features = X.shape[1]  # Количество признаков включая bias
weights = np.random.randn(n_features) * 0.01

print(f"\nНачальные веса: {weights}\n")

# Параметры градиентного спуска
learning_rate = 0.001  # Уменьшили с 0.01 до 0.001
n_epochs = 1000
m = X.shape[0]  # Количество объектов

# Для хранения истории потерь
loss_history = []

for epoch in range(n_epochs):
    # Предсказания модели
    y_pred = np.dot(X, weights)

    # Вычисление ошибки
    error = y_pred - y

    # Вычисление функции потерь (MSE)
    loss = np.mean(error ** 2)
    loss_history.append(loss)

    # Проверка на NaN
    if np.isnan(loss):
        print(f"Эпоха {epoch + 1}: Обнаружена NaN в значении потерь. Остановка обучения.")
        break

    # Вычисление градиентов
    gradients = (2 / m) * np.dot(X.T, error)

    # Проверка градиентов на NaN
    if np.isnan(gradients).any():
        print(f"Эпоха {epoch + 1}: Обнаружены NaN в градиентах. Остановка обучения.")
        break

    # Обновление весов
    weights -= learning_rate * gradients

    # Печать потерь каждые 100 эпох
    if (epoch + 1) % 100 == 0:
        print(f"Эпоха {epoch + 1}/{n_epochs}, Потеря: {loss:.4f}")

# Финальные веса
print(f"\nФинальные веса: {weights}\n")

# Визуализация функции потерь
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_history) + 1), loss_history, color='blue')
plt.title('График функции потерь (MSE) по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.grid(True)
plt.show()

# Предсказание окончательных значений
y_pred_final = np.dot(X, weights)

# Визуализация распределения целевой переменной: реальные vs предсказанные значения
plt.figure(figsize=(10, 6))
sns.histplot(y, bins=30, kde=True, color='skyblue', label='Реальные значения', stat="density")
sns.histplot(y_pred_final, bins=30, kde=True, color='salmon', label='Предсказанные значения', stat="density")
plt.title('Распределение целевой переменной: реальные vs предсказанные значения')
plt.xlabel('Значения целевой переменной')
plt.ylabel('Плотность')
plt.legend()
plt.show()

# Форматирование уравнения гиперплоскости
equation = "y = "
for i in range(n_features - 1):
    equation += f"{weights[i]:.4f} * {features[i]} + "
equation += f"{weights[-1]:.4f} * bias"

print(f"Уравнение полученной гиперплоскости:\n{equation}")

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Инициализируем модель линейной регрессии
model_sklearn = LinearRegression()

# Обучаем модель на тех же данных X и y
model_sklearn.fit(X, y)

# Предсказываем значения целевой переменной
y_pred_sklearn = model_sklearn.predict(X)

# Вычисляем среднеквадратичную ошибку (MSE)
mse_sklearn = mean_squared_error(y, y_pred_sklearn)
print(f"MSE модели scikit-learn: {mse_sklearn:.4f}")

# Извлекаем коэффициенты модели
coefficients_sklearn = model_sklearn.coef_

# Извлекаем смещение (bias) модели
intercept_sklearn = model_sklearn.intercept_

# Форматируем уравнение гиперплоскости
equation_sklearn = "y = "
for i in range(len(coefficients_sklearn) - 1):
    equation_sklearn += f"{coefficients_sklearn[i]:.4f} * {features[i]} + "
equation_sklearn += f"{coefficients_sklearn[-1]:.4f} * {features[-1]}"

print(f"\nУравнение гиперплоскости, полученное с помощью scikit-learn:\n{equation_sklearn}")

# Предполагается, что веса ручной модели сохранены в `weights_manual`
# Например:
# weights_manual = np.array([0.5679, -0.1235, ..., 1.2346])

# Извлекаем веса из ручной модели
weights_manual = weights  # Предполагается, что переменная `weights` содержит веса

# Извлекаем веса из модели scikit-learn
weights_sklearn = np.append(coefficients_sklearn, intercept_sklearn)

# Выводим оба набора весов
print("\nВесы модели, написанной своими руками:")
print(weights_manual)

print("\nВесы модели scikit-learn:")
print(weights_sklearn)

# Вычисляем разницу между весами
difference = weights_manual - weights_sklearn
print("\nРазница между весами (ручная модель - scikit-learn):")
print(difference)

# Визуализация распределения предсказанных значений
plt.figure(figsize=(12, 6))

# Гистограмма предсказаний ручной модели
sns.histplot(y_pred_final, bins=30, kde=True, color='salmon', label='Предсказанные значения (ручная модель)',
             stat="density")

# Гистограмма предсказаний scikit-learn модели
sns.histplot(y_pred_sklearn, bins=30, kde=True, color='green', label='Предсказанные значения (scikit-learn)',
             stat="density")

# Гистограмма реальных значений
sns.histplot(y, bins=30, kde=True, color='skyblue', label='Реальные значения', stat="density")

plt.title('Распределение целевой переменной: реальные vs предсказанные значения')
plt.xlabel('Значения целевой переменной')
plt.ylabel('Плотность')
plt.legend()
plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Инициализируем модель линейной регрессии
model_sklearn = LinearRegression()

# Обучаем модель на тех же данных X и y
model_sklearn.fit(X, y)

# Предсказываем значения целевой переменной
y_pred_sklearn = model_sklearn.predict(X)

# Вычисляем среднеквадратичную ошибку (MSE) для модели scikit-learn
mse_sklearn = mean_squared_error(y, y_pred_sklearn)

# Вычисляем коэффициент детерминации (R²) для модели scikit-learn
r2_sklearn = r2_score(y, y_pred_sklearn)

print(f"MSE модели scikit-learn: {mse_sklearn:.4f}")
print(f"Коэффициент детерминации (R²) модели scikit-learn: {r2_sklearn:.4f}\n")

# %%
# Вычисляем среднеквадратичную ошибку (MSE) для ручной модели
mse_manual = mean_squared_error(y, y_pred_final)

# Вычисляем коэффициент детерминации (R²) для ручной модели
r2_manual = r2_score(y, y_pred_final)

print(f"MSE ручной модели: {mse_manual:.4f}")
print(f"Коэффициент детерминации (R²) ручной модели: {r2_manual:.4f}\n")

# %%
# Выводим метрики обеих моделей
print("Оценка моделей линейной регрессии:\n")

print("Модель scikit-learn:")
print(f"  - MSE: {mse_sklearn:.4f}")
print(f"  - R²: {r2_sklearn:.4f}\n")

print("Ручная модель:")
print(f"  - MSE: {mse_manual:.4f}")
print(f"  - R²: {r2_manual:.4f}\n")

# %%
