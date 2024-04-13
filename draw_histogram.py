import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 讀取儲存的 y 值
with open('y_values_dump.pkl', 'rb') as file:
    y_values = pickle.load(file)

# 计数每个 y 值出现的次数
counts = Counter(y_values)

# 准备绘图数据
labels, values = zip(*counts.items())

# 绘制直方图
plt.bar(labels, values)
plt.xlabel('Y value')
plt.ylabel('Frequency')
plt.title('Histogram of Y Values')
plt.show()
