import matplotlib.pyplot as plt
import numpy as np

# Data for the Exponential Decay model
concentrations = ['20%', '40%', '60%', '78%', '91%']
exp_decay_r2 = [0.8129, 0.9486, 0.9929, 0.9758, 0.9432]
exp_decay_mse = [4.5602, 1.4719, 0.1032, 0.9707, 0.0628]
exp_decay_params = [
    [59.88, 0.00805, 51.85],
    [65.74, 0.01099, 52.94],
    [65.99, 0.00907, 50.02],
    [57.22, 0.00556, 41.75],
    [45.81, 0.00934, 48.02]
]

# Data for the Simplified Young-Lippmann model
young_lippmann_r2 = [0.9953, 0.9534, 0.9931, 0.9956, 1.0000]
young_lippmann_mse = [0.0690, 2.0758, 0.2680, 0.2131, 0.0000]
young_lippmann_params = [
    [74.72, 6.122e-6],
    [76.20, 7.169e-6],
    [75.38, 5.734e-6],
    [65.19, 1.719e-6],
    [69.40, 3.188e-5]
]

# Plotting Exponential Decay model results
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Plot R^2 and MSE for Exponential Decay model
axes[0].bar(concentrations, exp_decay_r2, color='blue', alpha=0.6, label='$R^2$')
axes[0].set_ylabel('$R^2$')
axes[0].set_ylim(0, 1.1)
axes[0].set_title('Exponential Decay Model')
axes[0].legend(loc='upper left')

ax2 = axes[0].twinx()
ax2.plot(concentrations, exp_decay_mse, color='red', marker='o', label='MSE')
ax2.set_ylabel('MSE')
ax2.legend(loc='upper right')

# Plot parameters for Exponential Decay model
param_labels = ['a', 'b', 'c']
for i, label in enumerate(param_labels):
    param_values = [params[i] for params in exp_decay_params]
    axes[1].plot(concentrations, param_values, marker='o', label=label)

axes[1].set_ylabel('Parameter Values')
axes[1].set_title('Exponential Decay Model Parameters')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("exp_decay_model_results.png")
plt.show()

# Plotting Simplified Young-Lippmann model results
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Plot R^2 and MSE for Simplified Young-Lippmann model
axes[0].bar(concentrations, young_lippmann_r2, color='blue', alpha=0.6, label='$R^2$')
axes[0].set_ylabel('$R^2$')
axes[0].set_ylim(0, 1.1)
axes[0].set_title('Simplified Young-Lippmann Model')
axes[0].legend(loc='upper left')

ax2 = axes[0].twinx()
ax2.plot(concentrations, young_lippmann_mse, color='red', marker='o', label='MSE')
ax2.set_ylabel('MSE')
ax2.legend(loc='upper right')

# Plot parameters for Simplified Young-Lippmann model
param_labels = ['theta_0', 'k']
for i, label in enumerate(param_labels):
    param_values = [params[i] for params in young_lippmann_params]
    axes[1].plot(concentrations, param_values, marker='o', label=label)

axes[1].set_ylabel('Parameter Values')
axes[1].set_title('Simplified Young-Lippmann Model Parameters')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("young_lippmann_model_results.png")
plt.show()
