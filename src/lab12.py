# Імпорт необхідних бібліотек
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ucimlrepo import fetch_ucirepo

# --- Етап 1: Завантаження даних ---
# Замініть 'ID_ДАТАСЕТУ' на ID обраного вами набору даних з UCI Machine Learning Repository.
# Посилання на репозиторій: https://archive.ics.uci.edu/ml/index.php
dataset = fetch_ucirepo(id=0) # Замініть 0 на ваш ID

# Отримання даних та цільової змінної
X = dataset.data.features
y = dataset.data.targets

# Створення єдиного DataFrame (якщо потрібно)
# Можливо, вам доведеться налаштувати цей крок залежно від структури вашого датасету
if y is not None:
    df = pd.concat([X, y], axis=1)
else:
    df = X

print("Назви стовпців:")
print(df.columns)

# --- Етап 2: Ознайомлення з даними ---

# Виведення метаданих
print("\nМетадані:")
print(dataset.metadata)

# Виведення інформації про змінні
print("\nІнформація про змінні:")
print(dataset.variables)

# Виведення перших 5 рядків датасету
print("\nПерші 5 рядків:")
print(df.head())

# Виведення основних статистичних характеристик
print("\nОписова статистика:")
print(df.describe())


# --- Етап 3: Побудова графіків ---

# Важливо: Замініть 'column_name', 'x_column', 'y_column' тощо на реальні назви стовпців з вашого датасету.

# 1. Гістограма
# Дозволяє побачити розподіл значень однієї числової змінної.
plt.figure(figsize=(10, 6))
sns.histplot(df['column_name'], bins=30, kde=True)
plt.title('Назва гістограми')
plt.xlabel('Назва осі X')
plt.ylabel('Частота')
plt.savefig('histogram.png')
# plt.show() # Розкоментуйте, якщо хочете побачити графік одразу

# 2. Діаграма розсіювання (Scatter plot)
# Показує залежність між двома числовими змінними.
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='x_column', y='y_column', hue='category_column') # hue - для розфарбовування за категоріями
plt.title('Назва діаграми розсіювання')
plt.xlabel('Назва осі X')
plt.ylabel('Назва осі Y')
plt.legend(title='Категорії')
plt.savefig('scatterplot.png')
# plt.show()

# 3. Стовпчикова діаграма (Bar plot)
# Порівнює середнє значення числової змінної для різних категорій.
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='category_column', y='value_column')
plt.title('Назва стовпчикової діаграми')
plt.xlabel('Категорії')
plt.ylabel('Середнє значення')
plt.xticks(rotation=45) # Поворот підписів на осі X для кращої читабельності
plt.savefig('barplot.png')
# plt.show()

# 4. Ящик з вусами (Box plot)
# Показує розподіл даних, медіану, квартилі та викиди.
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='category_column', y='value_column')
plt.title('Назва "Ящика з вусами"')
plt.xlabel('Категорії')
plt.ylabel('Значення')
plt.savefig('boxplot.png')
# plt.show()

# 5. Теплова карта кореляцій (Heatmap)
# Візуалізує матрицю кореляцій між числовими змінними.
plt.figure(figsize=(12, 8))
# Обчислюємо кореляційну матрицю тільки для числових стовпців
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Теплова карта кореляцій')
plt.savefig('heatmap.png')
# plt.show()

print("\nГрафіки збережено у файли .png")

# --- Етап 4: Заповнення звіту ---
# Не забудьте додати збережені графіки та ваші висновки у файл REPORT.md!