import os
import pickle
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt

title = input("請輸入圖片標題: ")


# Set up the root window for the file dialog
root = tk.Tk()
root.withdraw()  # Hide the root window

voltage_file_path = filedialog.askopenfilename(title="Select Voltage pkl File", filetypes=[("Pickle files", "*.pkl")])

if voltage_file_path:
    angle_file_path = filedialog.askopenfilename(title="Select Angle pkl File", filetypes=[("Pickle files", "*.pkl")])
    if angle_file_path:
        # Read the voltages
        with open(voltage_file_path, "rb") as voltage_file:
            voltages = pickle.load(voltage_file)
        
        # Read the angles
        with open(angle_file_path, "rb") as angle_file:
            angles_array = pickle.load(angle_file)
        
        angles_array = [[180 - float(angle) for angle in angles] for angles in angles_array]

        # 計算 means and 標準差
        left_angle_means = []
        left_angle_standard_deviations = []
        right_angle_means = []
        right_angle_standard_deviations = []
        for angles in angles_array:
            if len(angles) >= 6: # Make sure there are enough angles to perform calculation
                first_three_angles = angles[:3]
                last_three_angles = angles[-3:]

                # Calculate means
                left_angle_means.append(np.mean(first_three_angles))
                right_angle_means.append(np.mean(last_three_angles))

                # Calculate deviation
                left_angle_standard_deviations.append(np.std(first_three_angles))
                right_angle_standard_deviations.append(np.std(last_three_angles))

            else:
                raise Exception("有陣列不足六個角度，請檢查數據。")
        
        # 拆分數據用以繪圖
        max_voltage_index = voltages.index(max(voltages))
        # 切分 voltages 和 contact_angles 為兩個部分
        increasing_part_voltages = voltages[:max_voltage_index + 1]
        decreasing_part_voltages = voltages[max_voltage_index:]
        # left angle means
        increasing_part_left_angles_means = left_angle_means[:max_voltage_index + 1]
        decreasing_part_left_angles_means = left_angle_means[max_voltage_index:]
        # left angle standard deviations
        increasing_part_left_angle_standard_deviations = left_angle_standard_deviations[:max_voltage_index + 1]
        decreasing_part_left_angle_standard_deviations = left_angle_standard_deviations[max_voltage_index:]
        # right angle means
        increasing_part_right_angles_means = right_angle_means[:max_voltage_index + 1]
        decreasing_part_right_angles_means = right_angle_means[max_voltage_index:]
        # right angle standard deviations
        increasing_part_right_angle_standard_deviations = right_angle_standard_deviations[:max_voltage_index + 1]
        decreasing_part_right_angle_standard_deviations = right_angle_standard_deviations[max_voltage_index:]

        # 繪圖
        # 設定圖表中的字體大小
        plt.rcParams['font.size'] = 16  # 調整這裡的數值來改變字體大小
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['legend.fontsize'] = 16

        # 繪製左角度散點圖和誤差棒
        plt.errorbar(increasing_part_voltages, increasing_part_left_angles_means, 
                    yerr=increasing_part_left_angle_standard_deviations, 
                    fmt='o', label='Increasing Voltage - Left Angles', linestyle='-', capsize=3)
        plt.errorbar(decreasing_part_voltages, decreasing_part_left_angles_means, 
                    yerr=decreasing_part_left_angle_standard_deviations,
                    fmt="o", label="Decreasing Voltage - Left Angles", linestyle='-', capsize=3)

        # 繪製右角度散點圖和誤差棒
        plt.errorbar(increasing_part_voltages, increasing_part_right_angles_means, 
                    yerr=increasing_part_right_angle_standard_deviations, 
                    fmt='s', label='Increasing Voltage - Right Angles', linestyle='-', capsize=3)
        plt.errorbar(decreasing_part_voltages, decreasing_part_right_angles_means, 
                    yerr=decreasing_part_right_angle_standard_deviations,
                    fmt="s", label="Decreasing Voltage - Right Angles", linestyle='-', capsize=3)

        # 標題和軸標籤
        plt.title(title)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Angle (degrees)')

        # 顯示圖例
        plt.legend()

        # 顯示圖形
        plt.show()
        pass

    else:
        print("Doesn't choose a angle pkl file, app close.")

else:
    print("Doesn't choose a voltage pkl file, app close.")