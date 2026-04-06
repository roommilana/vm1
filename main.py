import numpy as np
import sympy as sm
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
from scipy.optimize import fsolve

print("задание 1") # создать матрицу 5x5 случайных целых чисел, принадлежащих полуотрезку [0, 10).
# Транспонировать. Вычислить ее определитель.

A=np.random.randint(0, 10, (5, 5))
print("Исходная матрица A:\n", A)
print("Транспонированная матрица A_T:\n", A.T)
print("Определитель A:", np.linalg.det(A))

print("\nзадание 2") #создать вектор-столбец и
#матрицу подходящих размеров. Выполнить умножение матриц.

vec=np.random.uniform(0,3,(1, 5)).T
B=np.random.randint(0, 10, (5, 5))
print("вектор-столбец v (5x1):\n", vec)
print("результат A * v (5x1):\n", A @ vec)
print("результат A * B:\n", A @ B)

print("\nзадание 3") #упростите выражение (2*x+3*y)**2-4*x*y*(x-y)/3
#и найдите его значение при x=1.038  и  y=sqrt(7)

x, y = sm.symbols('x, y')
expr = (2*x + 3*y)**2 - 4*x*y*(x - y)/3
expr_f=sm.simplify(expr)
val_x = 1.038
val_y = sm.sqrt(7)
val= expr_f.subs({x: val_x, y: val_y}).evalf()
#evalf() — вычисляет численное приближение (по умолчанию с 15 знаками)
print("выражение: \n", expr)
print("упрощенное выражение: \n", expr_f)
print(f"значение при x = {val_x}, y = {val_y} ≈ {val}")

print("\nзадание 4") #найдите частные производные от выражения из задания выше

diff_x = sm.diff(expr, x)
diff_y = sm.diff(expr, y)
print("производная по x =", sm.simplify(diff_x))
print("производная по y =", sm.simplify(diff_y))

print("\nзадание 5") #решить систему уравнений двумя способами
# Система:
# x1 - x3 = 1
# -x1 - x2 + 3x3 = -3
# x1 - 2x2 - 4x3 = 5

# Способ 1: NumPy (численно)
A_sys = np.array([[1, 0, -1], [-1, -1, 3], [1, -2, -4]], float)
B_sys = np.array([1, -3, 5], float)
X_np = np.linalg.solve(A_sys, B_sys)
print("решение (NumPy): x1 =", X_np[0], ", x2 =", X_np[1], ", x3 =", X_np[2])

# Способ 2: SymPy (символьно)
x1, x2, x3 = sm.symbols('x1, x2, x3')
eq1 = sm.Eq(x1 - x3, 1)
eq2 = sm.Eq(-x1 - x2 + 3*x3, -3)
eq3 = sm.Eq(x1 - 2*x2 - 4*x3, 5)
sol_sym = sm.solve((eq1, eq2, eq3), (x1, x2, x3))
print("решение (SymPy):", sol_sym)

print("\nзадание 6") #вычислить интеграл двумя способами
#sqrt(x)+3sqrt(x^2)

# 1. Символьное интегрирование (SymPy)
x_sym = sm.symbols('x')
f = sm.sqrt(x) + x**(sm.Rational(2, 3))
#SymPy: sp.integrate(подынтегральная_функция, (переменная, нижний_предел, верхний_предел))
integral_sym = sm.integrate(f, (x, 0, 1))
print("SymPy: определённый интеграл от 0 до 1 =", integral_sym.evalf())

# 2. Численное интегрирование (SciPy)
f_num = lambda x: np.sqrt(x) + x**(2/3)
#SciPy: integrate.quad(lambda x: ..., нижний_предел, верхний_предел)
integral_scipy, error = integrate.quad(f_num,0,1)
print("SciPy: определённый интеграл от 0 до 1 =", integral_scipy)

print("\nзадание 7") #вычислить интеграл двумя способами
x, y = sm.symbols('x y')
f2 = (x - y) * sm.exp(y)
inner = sm.integrate(f2, (x, 2*y, y)) # Внутренний интеграл по x
outer = sm.integrate(inner, (y, -1, 1)) # Внешний интеграл по y
print("двойной интеграл (SymPy):", outer.evalf())

# как в методичке
def f(x,y):
    return (x - y) * np.exp(y)
result, error = integrate.dblquad(f, -1, 1, lambda y: 2*y, lambda y: y)
print("SciPy (dblquad): результат =", result)
print("Оценка погрешности:", error)

print("\nзадание 8")
# Построить в одной системе координат графики функций: y = sin(x), y = sqrt(x + 5)
# Оси координат должны быть подписаны,
# графики должны быть разного цвета, должна быть выведена легенда.
# Точку пересечения (если она есть) выделить на графике оранжевым цветом.

# Функция для поиска корня: 3*sin(x) - sqrt(x+5) = 0
def equation(x):
    return 3 * np.sin(x) - np.sqrt(x + 5)

# Начальные приближения для двух корней (визуально на графике)
x_guess1 = 0.9    # между 0 и π/2
x_guess2 = 2.0    # между π/2 и π
x_guess3 = -3.6    # между -3π/2 и -π
# Численное нахождение корней
x1 = fsolve(equation, x_guess1)[0]
x2 = fsolve(equation, x_guess2)[0]
x3 = fsolve(equation, x_guess3)[0]
y1 = 3 * np.sin(x1)
y2 = 3 * np.sin(x2)
y3 = 3 * np.sin(x3)

# Создание объектов артборда и холста
# Построение графика
plt.figure(figsize=(8, 5), dpi=80)
ax = plt.subplot(111)

# Убираем правую и верхнюю границы
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Установить направление данных на координатной оси
 # 0 согласуется с нашей общей декартовой системой координат, 1 - противоположность
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))


# Подготовить данные, использовать распаковку последовательности

# X = np.linspace(-2*np.pi, 2*np.pi, 256, endpoint=True)
X = np.linspace(-5, 2*np.pi, 500, endpoint=True)
S, L = 3*np.sin(X), np.sqrt(X+5)

# Оранжевые точки пересечения (обе)
plt.scatter([x1, x2, x3], [y1, y2, y3], color='orange', s=80, zorder=5, label='Пересечения')

plt.plot(X, S, color="blue", linewidth=2.5, linestyle="-", label="Sin Function")
plt.plot(X, L, color="red", linewidth=2.5, linestyle="-", label="Lin Function")

plt.xlim(X.min() * 1.1, X.max() * 1.1)

# Изменить метку на оси координат
plt.xticks([-2*np.pi, -3*np.pi / 2, -np.pi, -np.pi / 2, 0, np.pi / 2, np.pi, 3*np.pi / 2, 2*np.pi ],
           [r'$-2\pi$', r'$-3\pi/2$', r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$', r'$+3 \pi/2$', r'$+ 2\pi$' ])

plt.ylim(S.min() * 1.1, S.max() * 1.1)
plt.yticks([-2, -1, +1, +2],
           [r'$-2$', r'$-1$', r'$+1$', r'$+2$'])

# #добавляем точку
# ax.scatter(x=105, y=110, c='g')

plt.legend(loc='upper left', frameon=False)
plt.grid()
plt.show()





