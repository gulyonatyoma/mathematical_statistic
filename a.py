import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import expon, skew
from statsmodels.distributions.empirical_distribution import ECDF

# Генерация выборки размера n = 25
np.random.seed(42)
n = 25
sample = np.random.exponential(scale=1, size=n)

# a) Мода, медиана, размах, коэффициент асимметрии
mode = sample[np.argmax(np.bincount(sample.astype(int)))]
median = np.median(sample)
range_ = np.ptp(sample)
skewness = skew(sample)

print(f"Mode: {mode:.2f}")
print(f"Median: {median:.2f}")
print(f"Range: {range_:.2f}")
print(f"Skewness: {skewness:.2f}")

# b) Эмпирическая функция распределения, гистограмма и boxplot
plt.figure(figsize=(15, 5))

# Эмпирическая функция распределения
plt.subplot(1, 3, 1)
ecdf = ECDF(sample)
plt.step(ecdf.x, ecdf.y, where='post')
plt.title('Empirical CDF')

# Гистограмма
plt.subplot(1, 3, 2)
sns.histplot(sample, bins=10, kde=True)
plt.title('Histogram')

# Boxplot
plt.subplot(1, 3, 3)
sns.boxplot(sample)
plt.title('Boxplot')

plt.show()

# c) Оценка плотности по ЦПТ и бутстрапу
means = [np.mean(np.random.choice(sample, size=n, replace=True)) for _ in range(1000)]

plt.figure(figsize=(10, 5))
sns.kdeplot(means, label='Bootstrap Mean Density')
sns.kdeplot(np.random.normal(np.mean(sample), np.std(sample)/np.sqrt(n), size=1000), label='CLT Density')
plt.legend()
plt.title('Comparison of Bootstrap and CLT Densities')
plt.show()

# d) Бутстрап для коэффициента асимметрии
bootstrap_skewness = [skew(np.random.choice(sample, size=n, replace=True)) for _ in range(1000)]

plt.figure(figsize=(10, 5))
sns.kdeplot(bootstrap_skewness)
plt.title('Bootstrap Skewness Density')
plt.show()

# e) Вероятность, что коэффициент асимметрии < 1
probability = np.mean(np.array(bootstrap_skewness) < 1)
print(f"P(skewness < 1): {probability:.2f}")

# f) Бутстраповская оценка плотности распределения медианы выборки
bootstrap_medians = [np.median(np.random.choice(sample, size=n, replace=True)) for _ in range(1000)]

plt.figure(figsize=(10, 5))
sns.kdeplot(bootstrap_medians)
plt.title('Bootstrap Density Estimate of Sample Medians')
plt.show()
