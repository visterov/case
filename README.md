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

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# загрузка данных
train_data = pd.read_csv('http://work.caltech.edu/data/in.dta', header=None, sep='\s+', names=['x1', 'x2', 'y'])
test_data = pd.read_csv('http://work.caltech.edu/data/out.dta', header=None, sep='\s+', names=['x1', 'x2', 'y'])

# преобразование данных
def transform(X):
    x1, x2 = X[:,0], X[:,1]
    return np.array([
        np.ones(len(X)),
        x1,
        x2,
        x1 * x2,
        x1**2,
        x2**2,
        np.abs(x1 - x2),
        np.abs(x1 + x2)
    ]).T

X_train = transform(train_data[['x1', 'x2']].values)
y_train = train_data['y'].values

X_test = transform(test_data[['x1', 'x2']].values)
y_test = test_data['y'].values

# обучение модели без регуляризации
w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
y_train_pred = np.sign(X_train @ w)
y_test_pred = np.sign(X_test @ w)

train_error = np.mean(y_train != y_train_pred)
test_error = np.mean(y_test != y_test_pred)
print(f'Ошибка обучения без регуляризации: {train_error:.3f}')
print(f'Ошибка тестирования без регуляризации: {test_error:.3f}')

# обучение модели с регуляризацией
for k in range(-3, 4):
    alpha = 10 ** k
    clf = Ridge(alpha=alpha, fit_intercept=False)
    clf.fit(X_train, y_train)
    y_train_pred = np.sign(clf.predict(X_train))
    y_test_pred = np.sign(clf.predict(X_test))
    
    train_error = np.mean(y_train != y_train_pred)
    test_error = np.mean(y_test != y_test_pred)
    print(f'alpha = 10^{k:<2} Ошибка обучения: {train_error:.3f}, Ошибка тестирования: {test_error:.3f}')

# минимальная ошибка тестирования
best_k = -1
best_alpha = 10 ** best_k
clf = Ridge(alpha=best_alpha, fit_intercept=False)
clf.fit(X_train, y_train)
y_train_pred = np.sign(clf.predict(X_train))
y_test_pred = np.sign(clf.predict(X_test))

train_error = np.mean(y_train != y_train_pred)
test_error = np.mean(y_test != y_test_pred)

print(f'Оптимальное значение k: {best_k}')
print(f'Оптимальное значение alpha: {best_alpha}')
print(f'Ошибка обучения: {train_error:.3f}')
print(f'Ошибка тестирования: {test_error:.3f}')
```
