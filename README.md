# Кейс №6

Условие:

В этом кейсе используйте данные, представленные в файлах:
`http://work.caltech.edu/data/in.dta`  — данные для обучения,
`http://work.caltech.edu/data/out.dta` — данные для тестирования.
Каждая строка файла соответствует точке x=(x_1,x_2), так что X = R^2, за которой следует соответствующая метка из Y ={-1; 1}. 


1. Обучите модель линейной регрессии на обучающем наборе после выполнения нелинейного преобразования
a(x_1,x_2)=(1,x_1,x_2,x_1 x_2,〖x_1〗^2,〖x_2〗^2,|x_1  - x_2 |,|x_1  + x_2 |).


2. Рассчитайте ошибку обучения и ошибку тестирования для обученной модели (ошибка классификации рассчитывается как среднее число неверно классифицированных объектов).


3. Добавьте в линейную регрессию регуляризацию на основе метода сокращения весов, то есть к квадрату ошибки линейной регрессии  добавьте слагаемое λ∑_("i\="0)^d▒w_i^2 , где d - размерность вектора данных.


4. Обучите модель линейной регрессии с регуляризацией для λ"\=" 〖"\1\0"〗^k "\," k"\="-"\3\,"-"\2\,"-"\1\,\0\,\1\,\2\,\3" .


5. Для каждого значения k рассчитайте ошибку обучения и тестирования.


6. Для какого значения k ошибка тестирования минимальная?

---


Решение:

Чтение данных из файлов:

```python
import pandas as pd
# чтение данных для обучения
data_train = pd.read_csv('http://work.caltech.edu/data/in.dta', header=None, delimiter='\t')
X_train = data_train.iloc[:, :-1].values
y_train = data_train.iloc[:, -1].values
# чтение данных для тестирования
data_test = pd.read_csv('http://work.caltech.edu/data/out.dta', header=None, delimiter='\t')
X_test = data_test.iloc[:, :-1].values
y_test = data_test.iloc[:, -1].values
```

Нелинейное преобразование и обучение модели линейной регрессии:

```python
import numpy as np
from sklearn.linear_model import LinearRegression
# Нелинейное преобразование
def transform(X):
    """
    Функция осуществляет нелинейное преобразование данных X
    """
    N = X.shape[0]
    Z = np.zeros((N, 8))
    Z[:, 0] = np.ones(N)
    Z[:, 1:3] = X
    Z[:, 3] = X[:, 0] * X[:, 1]
    Z[:, 4] = X[:, 0] ** 2
    Z[:, 5] = X[:, 1] ** 2
    Z[:, 6] = np.abs(X[:, 0] - X[:, 1])
    Z[:, 7] = np.abs(X[:, 0] + X[:, 1])
    return Z
# Нелинейное преобразование данных обучающего и тестового набора
Z_train = transform(X_train)
Z_test = transform(X_test)
# Обучение модели линейной регрессии
reg = LinearRegression().fit(Z_train, y_train)
```

Рассчитаем ошибку на обучающем наборе:

```python
# Предсказание на обучающем наборе
y_train_pred = np.sign(reg.predict(Z_train))
# Расчет ошибки на обучающем наборе
train_error = np.mean(y_train != y_train_pred)
print('Ошибка на обучающем наборе:', train_error)
```

Рассчитаем ошибку на тестовом наборе:

```python
# Предсказание на тестовом наборе
y_test_pred = np.sign(reg.predict(Z_test))
# Расчет ошибки на тестовом наборе
test_error = np.mean(y_test != y_test_pred)
print('Ошибка на тестовом наборе:', test_error)
```

Внесем изменения в код для добавления L2-регуляризации в модель линейной регрессии:

```python
from sklearn.linear_model import Ridge
# Создание модели с регуляризацией L2
reg = Ridge(alpha=1.0)
# Обучение модели на обучающем наборе
reg.fit(Z_train, y_train)
# Предсказание на обучающем наборе
y_train_pred = np.sign(reg.predict(Z_train))
# Расчет ошибки на обучающем наборе
train_error = np.mean(y_train != y_train_pred)
print('Ошибка на обучающем наборе:', train_error)
# Предсказание на тестовом наборе
y_test_pred = np.sign(reg.predict(Z_test))
# Расчет ошибки на тестовом наборе
test_error = np.mean(y_test != y_test_pred)
print('Ошибка на тестовом наборе:', test_error)
```

Воспользуемся циклом, чтобы обучить модели с различными значениями коэффициента регуляризации:

```python
for k in range(-3, 4):
    alpha = 10**k
    reg = Ridge(alpha=alpha)
    reg.fit(Z_train, y_train)
    y_train_pred = np.sign(reg.predict(Z_train))
    train_error = np.mean(y_train != y_train_pred)
    y_test_pred = np.sign(reg.predict(Z_test))
    test_error = np.mean(y_test != y_test_pred)
    print(f"Коэффициент регуляризации: {alpha:.0e}, Ошибка на обучающем наборе: {train_error:.3f}, Ошибка на тестовом наборе: {test_error:.3f}")
```

Здесь мы использовали функцию range для создания последовательности значений k от -3 до 3. Затем мы использовали форматирование строк f-строками, чтобы вывести значения коэффициента регуляризации и ошибок на обучающем и тестовом наборах для каждой модели.

Создадим списки для ошибок на обучающем и тестовом наборах и сохраним значения в эти списки в каждой итерации цикла:

```python
train_errors = []
test_errors = []
for k in range(-3, 4):
    alpha = 10**k
    reg = Ridge(alpha=alpha)
    reg.fit(Z_train, y_train)
    y_train_pred = np.sign(reg.predict(Z_train))
    train_error = np.mean(y_train != y_train_pred)
    y_test_pred = np.sign(reg.predict(Z_test))
    test_error = np.mean(y_test != y_test_pred)
    train_errors.append(train_error)
    test_errors.append(test_error)
    print(f"Коэффициент регуляризации: {alpha:.0e}, Ошибка на обучающем наборе: {train_error:.3f}, Ошибка на тестовом наборе: {test_error:.3f}")
```

Мы можем найти значение k, при котором ошибка тестирования минимальная, из списка test_errors, который мы создали на предыдущем шаге. Для этого мы можем использовать метод index для поиска индекса элемента списка с минимальным значением:

```python
best_k = np.argmin(test_errors) - 3
print(f"Оптимальное значение k: {best_k}, минимальная ошибка на тестовом наборе: {test_errors[best_k+3]:.3f}")
```
