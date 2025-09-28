# Домашка 4
# Тема: sklearn, custom dataset generation, linear regression, non-linear regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ==================================================
# 1. Лінійний датасет (y = 3x + 4 + шум)
# ==================================================
np.random.seed(42)
X_lin = 2 * np.random.rand(100, 1)
y_lin = 3 * X_lin + 4 + np.random.randn(100, 1)

# Лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X_lin, y_lin)
y_pred_lin = lin_reg.predict(X_lin)

# Вивід параметрів
print("Лінійна регресія")
print("Коефіцієнт:", lin_reg.coef_)
print("Вільний член:", lin_reg.intercept_)

# Візуалізація
plt.scatter(X_lin, y_lin, color="blue", label="Дані")
plt.plot(X_lin, y_pred_lin, color="black", linewidth=2, label="Модель")
plt.title("Linear Regression")
plt.legend()
plt.show()


# ==================================================
# 2. Нелінійний датасет (y = x^2 + шум)
# ==================================================
X_nonlin = np.linspace(-3, 3, 100).reshape(-1, 1)
y_nonlin = X_nonlin**2 + np.random.randn(100, 1) * 2

# Поліноміальна регресія степеня 2
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_nonlin, y_nonlin)
y_pred_nonlin = poly_model.predict(X_nonlin)

print("\nНелінійна регресія (поліном степеня 2)")

# Візуалізація
plt.scatter(X_nonlin, y_nonlin, color="red", label="Дані")
plt.plot(X_nonlin, y_pred_nonlin, color="black", linewidth=2, label="Модель")
plt.title("Non-linear Regression (Polynomial)")
plt.legend()
plt.show()
