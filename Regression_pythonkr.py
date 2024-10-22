# Импорт основных библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Импорт библиотек для машинного обучения
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Настройка стиля графиков
sns.set(style='whitegrid')
%matplotlib inline

#%%
# Загрузка датасета "wine-quality-red" из OpenML
dataset = fetch_openml(name='wine-quality-red', version=1, as_frame=True)

# Разделение данных на признаки (X) и целевую переменную (y)
X = dataset.data
y = dataset.target

# Преобразование целевой переменной в числовой тип
y = y.astype(float)

# Просмотр первых пяти строк признаков
print("Первые 5 строк признаков (X):")
display(X.head())

# Просмотр первых пяти значений целевой переменной
print("\nПервые 5 значений целевой переменной (y):")
print(y.head())


#%%
# Проверка на пропущенные значения в признаках
print("Количество пропущенных значений в признаках (X):")
print(X.isnull().sum())

# Проверка на пропущенные значения в целевой переменной
print("\nКоличество пропущенных значений в целевой переменной (y):")
print(y.isnull().sum())

#%%
# Инициализация стандартизатора
scaler = StandardScaler()

# Масштабирование признаков
X_scaled = scaler.fit_transform(X)

# Преобразование обратно в DataFrame для удобства
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Просмотр первых пяти строк масштабированных признаков
print("Признаки после масштабирования (первые 5 строк):")
display(X_scaled.head())

#%%
# Добавление столбца единиц для смещения (bias)
X_scaled['bias'] = 1

# Преобразование DataFrame в numpy массив
X_numpy = X_scaled.values
y_numpy = y.values.reshape(-1, 1)  # Преобразование y в столбец

print(f"Форма X после добавления bias: {X_numpy.shape}")
print(f"Форма y: {y_numpy.shape}")

#%%
# Разделение данных на обучающую и тестовую выборки (80% - обучение, 20% - тест)
X_train, X_test, y_train, y_test = train_test_split(X_numpy, y_numpy, test_size=0.2, random_state=42)

print(f"Форма обучающей выборки X_train: {X_train.shape}")
print(f"Форма тестовой выборки X_test: {X_test.shape}")
print(f"Форма обучающей выборки y_train: {y_train.shape}")
print(f"Форма тестовой выборки y_test: {y_test.shape}")

#%%
# Количество признаков (включая bias)
n_features = X_train.shape[1]

# Инициализация весов нулями
weights_manual = np.zeros((n_features, 1))

# Параметры градиентного спуска
learning_rate = 0.01
n_epochs = 1000
m_train = X_train.shape[0]

# Список для хранения истории потерь
loss_history = []

#%%
for epoch in range(n_epochs):
    # 1. Предсказание
    y_pred = np.dot(X_train, weights_manual)

    # 2. Вычисление ошибки
    error = y_train - y_pred

    # 3. Вычисление функции потерь (MSE)
    loss = (1/m_train) * np.sum(error ** 2)
    loss_history.append(loss)

    # 4. Вычисление градиентов
    gradients = (-2/m_train) * np.dot(X_train.T, error)

    # 5. Обновление весов
    weights_manual = weights_manual - learning_rate * gradients

    # 6. Вывод потерь каждые 100 эпох
    if (epoch + 1) % 100 == 0:
        print(f"Эпоха {epoch+1}, MSE: {loss:.4f}")

# 7.5. Формирование уравнения гиперплоскости для ручной модели

# Получение списка признаков (включая bias)
features = list(X_scaled.columns)

# Формирование уравнения гиперплоскости
equation_manual = "y = "
for i in range(len(features)):
    equation_manual += f"{weights_manual[i][0]:.4f} * {features[i]} + "
# Удаление последнего знака '+' и пробела
equation_manual = equation_manual.rstrip(' + ')

print(f"\nУравнение гиперплоскости (ручная модель):\n{equation_manual}")


#%%
# Визуализация функции потерь (MSE) по эпохам
plt.figure(figsize=(10, 6))
plt.plot(range(n_epochs), loss_history, color='blue')
plt.title('Функция потерь (MSE) по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.grid(True)
plt.show()

#%%
# Предсказание на тестовой выборке
y_pred_manual = np.dot(X_test, weights_manual)

# Вычисление метрик
mse_manual = mean_squared_error(y_test, y_pred_manual)
r2_manual = r2_score(y_test, y_pred_manual)

print(f"Среднеквадратичная ошибка (MSE) ручной модели: {mse_manual:.4f}")
print(f"Коэффициент детерминации (R²) ручной модели: {r2_manual:.4f}")

#%%
# Инициализация модели линейной регрессии с отключённым встроенным смещением
model_sklearn = LinearRegression(fit_intercept=False)

# Обучение модели на обучающей выборке
model_sklearn.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred_sklearn = model_sklearn.predict(X_test)

# Вычисление метрик
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)

print(f"Среднеквадратичная ошибка (MSE) модели scikit-learn: {mse_sklearn:.4f}")
print(f"Коэффициент детерминации (R²) модели scikit-learn: {r2_sklearn:.4f}")

#%%
# Извлечение коэффициентов модели
coefficients_sklearn = model_sklearn.coef_.flatten()

# Извлечение смещения (bias) модели
intercept_sklearn = model_sklearn.intercept_

# Проверка intercept (должен быть 0, так как fit_intercept=False)
print(f"Intercept модели scikit-learn: {intercept_sklearn:.4f}")  # Должен быть 0

# Форматирование уравнения гиперплоскости
features = list(X_scaled.columns)
equation_sklearn = "y = "
for i in range(len(coefficients_sklearn)):
    equation_sklearn += f"{coefficients_sklearn[i]:.4f} * {features[i]} + "
# Удаление последнего знака '+' и пробела
equation_sklearn = equation_sklearn.rstrip(' + ')

print(f"\nУравнение гиперплоскости, полученное с помощью scikit-learn:\n{equation_sklearn}")


#%%
# Извлечение весов из ручной модели
weights_manual_flat = weights_manual.flatten()

# Извлечение весов из модели scikit-learn
weights_sklearn = coefficients_sklearn  # bias уже включён

# Вывод весов ручной модели
print("\nВесы модели, написанной своими руками:")
print(weights_manual_flat)

# Вывод весов модели scikit-learn
print("\nВесы модели scikit-learn:")
print(weights_sklearn)

# Вычисление разницы между весами
difference = weights_manual_flat - weights_sklearn
print("\nРазница между весами (ручная модель - scikit-learn):")
print(difference)

#%%
# Визуализация распределения предсказанных значений
plt.figure(figsize=(12, 6))

# Гистограмма предсказаний ручной модели
sns.histplot(y_pred_manual, bins=30, kde=True, color='salmon', label='Предсказанные значения (ручная модель)', stat="density")

# Гистограмма предсказаний scikit-learn модели
sns.histplot(y_pred_sklearn, bins=30, kde=True, color='green', label='Предсказанные значения (scikit-learn)', stat="density")

# Гистограмма реальных значений
sns.histplot(y_test, bins=30, kde=True, color='skyblue', label='Реальные значения', stat="density")

plt.title('Распределение целевой переменной: реальные vs предсказанные значения')
plt.xlabel('Значения целевой переменной')
plt.ylabel('Плотность')
plt.legend()
plt.show()

#%%
# Создание сводной таблицы метрик
metrics = pd.DataFrame({
    'Модель': ['Ручная модель', 'scikit-learn'],
    'MSE': [mse_manual, mse_sklearn],
    'R²': [r2_manual, r2_sklearn]
})

display(metrics)
