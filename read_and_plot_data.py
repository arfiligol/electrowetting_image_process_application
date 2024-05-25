import tkinter as tk
from tkinter import filedialog
import pickle
import matplotlib.pyplot as plt

def load_and_plot_data(title):
    # Set up the root window for the file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open the file dialog to select a pkl file
    file_path = filedialog.askopenfilename(title="Select a PKL file", filetypes=[("Pickle files", "*.pkl")])
    if not file_path:
        print("No file selected.")
        return

    # Load the data from the selected pkl file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Unpack the list of tuples into two separate lists for plotting
    voltages, contact_angles = zip(*data)
    # 找到最大電壓的索引
    max_voltage_index = voltages.index(max(voltages))

    # 切分 voltages 和 contact_angles 為兩個部分
    first_part_voltages = voltages[:max_voltage_index + 1]
    second_part_voltages = voltages[max_voltage_index:]
    first_part_contact_angles = contact_angles[:max_voltage_index + 1]
    second_part_contact_angles = contact_angles[max_voltage_index:]

    # 接觸角度取負號
    first_part_contact_angles = [-angle for angle in first_part_contact_angles]
    second_part_contact_angles = [-angle for angle in second_part_contact_angles]

    # 合併兩部分資料
    # first_part_data = list(zip(first_part_voltages, first_part_contact_angles))ㄔㄣ
    # second_part_data = list(zip(second_part_voltages, second_part_contact_angles))


    # Plot the data using Matplotlib
    # Set rcParams (need to be set before plot since they are global variables)
    plt.rcParams['font.size'] = 30
    plt.rcParams['axes.labelsize'] = 30
    plt.rcParams['axes.titlesize'] = 34
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['legend.fontsize'] = 24


    plt.figure(figsize=(11, 8))

    plt.plot(first_part_voltages, first_part_contact_angles, marker = "o", markersize = 15, linestyle = "--", linewidth = 5, label='Adding Voltage')
    plt.plot(second_part_voltages, second_part_contact_angles, marker = "^", markersize = 15, linestyle = "--", linewidth = 5, label="Decresing Voltage")
    
    plt.xlabel('Voltage (V)')
    plt.ylabel('Contact Angle (degrees)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    


    plt.show()

#
title = input("請輸入圖片標題: ")

# Call the function to execute the file selection and plotting
load_and_plot_data(title)
