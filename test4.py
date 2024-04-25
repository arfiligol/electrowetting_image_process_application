import matplotlib.pyplot as plt
import numpy as np

# 橢圓參數
center = (116.45687866210938, 243.03634643554688)
axes = (175.7816162109375, 75.53372192382812)  # MA, ma
angle = 41.7218  # 旋轉角度

# 產生橢圓的點
theta = np.linspace(0, 2 * np.pi, 100)
x = axes[0] * np.cos(theta)
y = axes[1] * np.sin(theta)

# 旋轉橢圓
x_rotated = x * np.cos(np.radians(angle)) - y * np.sin(np.radians(angle))
y_rotated = x * np.sin(np.radians(angle)) + y * np.cos(np.radians(angle))

# 移動橢圓到正確的中心位置
x_final = x_rotated + center[0]
y_final = y_rotated + center[1]

# 繪圖
plt.figure(figsize=(8, 6))
plt.plot(x_final, y_final, 'b')  # 繪製藍色橢圓
plt.scatter(center[0], center[1], color='red')  # 標記橢圓中心
plt.title('Ellipse with Rotation')
plt.xlim(center[0] - 200, center[0] + 200)
plt.ylim(center[1] - 200, center[1] + 200)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()
