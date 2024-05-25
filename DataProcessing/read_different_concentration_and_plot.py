import os
import pickle
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt

title = input("Please enter the image title: ")

def plot_handtake_angle_and_voltage(voltages, angles_array, dataset_number, total_datasets, legend_tag):
    # Calculate means and standard deviations
    left_angle_means = []
    left_angle_standard_deviations = []
    right_angle_means = []
    right_angle_standard_deviations = []
    for angles in angles_array:
        if len(angles) >= 6:  # Ensure there are enough angles to perform the calculation
            first_three_angles = angles[:3]
            last_three_angles = angles[-3:]

            # Calculate means
            left_angle_means.append(np.mean(first_three_angles))
            right_angle_means.append(np.mean(last_three_angles))

            # Calculate standard deviations
            left_angle_standard_deviations.append(np.std(first_three_angles))
            right_angle_standard_deviations.append(np.std(last_three_angles))
        else:
            raise Exception("Some arrays have fewer than six angles; please check the data.")

    # Split data for plotting
    max_voltage_index = voltages.index(max(voltages))
    # Split voltages and contact_angles into two parts
    increasing_part_voltages = voltages[:max_voltage_index + 1]
    decreasing_part_voltages = voltages[max_voltage_index:]
    # Left angle means
    increasing_part_left_angles_means = left_angle_means[:max_voltage_index + 1]
    decreasing_part_left_angles_means = left_angle_means[max_voltage_index:]
    # Left angle standard deviations
    increasing_part_left_angle_standard_deviations = left_angle_standard_deviations[:max_voltage_index + 1]
    decreasing_part_left_angle_standard_deviations = left_angle_standard_deviations[max_voltage_index:]
    # Right angle means
    increasing_part_right_angles_means = right_angle_means[:max_voltage_index + 1]
    decreasing_part_right_angles_means = right_angle_means[max_voltage_index:]
    # Right angle standard deviations
    increasing_part_right_angle_standard_deviations = right_angle_standard_deviations[:max_voltage_index + 1]
    decreasing_part_right_angle_standard_deviations = right_angle_standard_deviations[max_voltage_index:]

    # Plotting
    # Set font sizes in the chart
    plt.rcParams['font.size'] = 32
    plt.rcParams['axes.labelsize'] = 32
    plt.rcParams['axes.titlesize'] = 36
    plt.rcParams['xtick.labelsize'] = 28
    plt.rcParams['ytick.labelsize'] = 28
    plt.rcParams['legend.fontsize'] = 24

    # Define the color map and select colors based on the dataset number
    cmap = plt.cm.get_cmap('viridis', total_datasets * 2)  # More colors if there are more datasets

    # Calculate the color indices
    color_index_increase = dataset_number * 2
    color_index_decrease = dataset_number * 2 + 1

    # Plot left angles scatter plot and error bars
    plt.errorbar(increasing_part_voltages, increasing_part_left_angles_means,
                 yerr=increasing_part_left_angle_standard_deviations,
                 fmt='o', label=f'Increasing Voltage - Left Angles (Dataset {legend_tag})',
                 linestyle='-', capsize=3, color=cmap(color_index_increase))
    plt.errorbar(decreasing_part_voltages, decreasing_part_left_angles_means,
                 yerr=decreasing_part_left_angle_standard_deviations,
                 fmt="o", label=f"Decreasing Voltage - Left Angles (Dataset {legend_tag})",
                 linestyle='-', capsize=3, color=cmap(color_index_decrease))

    # Plot right angles scatter plot and error bars
    plt.errorbar(increasing_part_voltages, increasing_part_right_angles_means,
                 yerr=increasing_part_right_angle_standard_deviations,
                 fmt='s', label=f'Increasing Voltage - Right Angles (Dataset {legend_tag})',
                 linestyle='-', capsize=3, color=cmap(color_index_increase))
    plt.errorbar(decreasing_part_voltages, decreasing_part_right_angles_means,
                 yerr=decreasing_part_right_angle_standard_deviations,
                 fmt="s", label=f"Decreasing Voltage - Right Angles (Dataset {legend_tag})",
                 linestyle='-', capsize=3, color=cmap(color_index_decrease))

# Load and process data for each dataset (Handtake)
datasets = ["4mm - Glycerin 20% - 2", "4mm - Glycerin 40% - 2"]
for i, file_name in enumerate(datasets):
    volt_filename = f"data/txt/{file_name}.pkl"
    angle_filename = f"hand_record_data/{file_name}.pkl"

    with open(volt_filename, "rb") as volt_file:
        c_vol = pickle.load(volt_file)

    with open(angle_filename, "rb") as ang_file:
        c_angle = pickle.load(ang_file)
        c_angle = [[180 - float(angle) for angle in angles] for angles in c_angle]

    legend_tag = file_name.split("Glycerin ")[1]
    plot_handtake_angle_and_voltage(c_vol, c_angle, i, len(datasets), legend_tag)

# Load 

# Show the plot
plt.title(title)
plt.xlabel("Voltage (V)")
plt.ylabel("Angle (Degrees)")
plt.legend()
plt.show()
