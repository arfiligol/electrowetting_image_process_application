import numpy as np
import matplotlib.pyplot as plt

# Constants for the simulation
cos_theta_Y = 0  # Given cos(theta_Y) is 0
d = 11e-6  # Thickness d is 11 micrometers
epsilon_0 = 8.854e-12  # Permittivity of free space in F/m
epsilon_d = 2.2  # Relative permittivity
sigma_lv = 72e-3  # Surface tension of water in N/m at room temperature

# Range of applied voltages in volts (up to a reasonable maximum voltage for electrowetting)
voltages = np.linspace(0, 1000, 10000)  # 0 to 100 volts

# Calculate cos(theta) using the given formula for the range of voltages
cos_theta = cos_theta_Y + (epsilon_0 * epsilon_d * voltages**2) / (2 * sigma_lv * d)
theta = np.arccos(cos_theta) * 180 / 3.14
# Plot the result
plt.figure(figsize=(10, 5))
plt.plot(voltages, theta, label='$\\theta$')
plt.title('Electrowetting Simulation')
plt.xlabel('Applied Voltage (V)')
plt.ylabel('$\\theta$ (degree)')
plt.grid(True)
plt.legend()
plt.show()
