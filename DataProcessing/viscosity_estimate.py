import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Provided data for viscosity at 20°C and 30°C
concentration = [20, 40, 60, 78, 91]  # concentration in %
viscosity_20C = [1.84, 3.63, 11.67, 48.79, 220.52]  # viscosity in mPa.s at 20°C
viscosity_30C = [1.47, 2.83, 7.95, 30.32, 125.89]  # viscosity in mPa.s at 30°C

# Assuming linear interpolation to estimate viscosity at 25°C
viscosity_25C = []

for v20, v30 in zip(viscosity_20C, viscosity_30C):
    viscosity = v20 + (v30 - v20) * (25 - 20) / (30 - 20)
    viscosity_25C.append(viscosity)

# DataFrame to show the result
data = {
    "Concentration (%)": concentration,
    "Viscosity at 20°C (mPa.s)": viscosity_20C,
    "Viscosity at 30°C (mPa.s)": viscosity_30C,
    "Estimated Viscosity at 25°C (mPa.s)": viscosity_25C,
}

df = pd.DataFrame(data)
# import ace_tools as tools; tools.display_dataframe_to_user(name="Estimated Viscosity at 25°C", dataframe=df)

# Plot the diagram
# Set rcParams (need to be set before plot since they are global variables)
plt.rcParams['font.size'] = 32
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 34
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 24
# Plotting the viscosity at different temperatures
plt.figure(figsize=(12, 8))
plt.plot(concentration, viscosity_20C, label="Viscosity at 20°C", marker='o', markersize=15, linewidth=5)
plt.plot(concentration, viscosity_30C, label="Viscosity at 30°C", marker='o', markersize=15, linewidth=5)
plt.plot(concentration, viscosity_25C, label="Estimated Viscosity at 25°C", marker='o', linestyle='--', markersize=15, linewidth=5)
plt.xlabel("Concentration (%)")
plt.ylabel("Viscosity (mPa.s)")
plt.title("Viscosity vs Concentration \nat Different Temperatures")
plt.legend()
plt.grid(True)
plt.show()
