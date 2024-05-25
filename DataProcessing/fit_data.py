import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
from tkinter import Tk, simpledialog
from tkinter import filedialog
from sklearn.metrics import r2_score, mean_squared_error

def young_lippmann_model(voltage, theta_0, epsilon_0, epsilon_r, gamma_LG, d):
    return np.arccos(np.cos(theta_0) + (epsilon_0 * epsilon_r * (voltage ** 2)) / (2 * gamma_LG * d)) * 180 / 3.14

def simplified_young_lippmann_model(voltage, theta_0, k):
    return np.arccos(np.cos(theta_0) + k * (voltage**2)) * 180 / np.pi

def exponential_model(voltage, a, b, c):
    return a * np.exp(-b * voltage) + c

def main(title):
    root = Tk()
    root.withdraw() # Hide the main window

    data_file_path = filedialog.askopenfilename(title="選擇要 Fitting 的 data", filetypes=[("Pickle File", "*.pkl")])
    if data_file_path:
        with open(data_file_path, "rb") as data_file:
            data = pickle.load(data_file)

        # Only pick some part of the data
        voltage_data, angle_data = zip(*data)
        voltage_data = np.array(voltage_data)
        angle_data = np.array(angle_data) 
        # Make angle values positive
        angle_data = np.abs(angle_data)
    else:
        print("Quit: User did not pick a file to fit.")
        return

    # Set rcParams (need to be set before plot since they are global variables)
    plt.rcParams['font.size'] = 32
    plt.rcParams['axes.labelsize'] = 32
    plt.rcParams['axes.titlesize'] = 36
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 24

    # Draw the Contact Angle v.s Voltage
    plt.figure(figsize=(13, 10))

    # Determine the index where voltage is less than 'voltage_threshold' and only
    min_voltage_for_exp = simpledialog.askfloat("Input", "Enter a minimum voltage for exponential decay fitting range.")
    if min_voltage_for_exp is None:
        print("Minimum voltage for exponential decay fitting range not provided. Exiting the program.")
        return
    max_voltage_for_young = simpledialog.askfloat("Input", "Enter a maximum voltage for simplified Young-Lippmann fitting range.")
    if max_voltage_for_young is None:
        print("Maximum voltage for simplified Young-Lippmann fitting range not provided. Exiting the program.")
        return
    print(f"Minimum voltage for Exponential Decay fitting range: {min_voltage_for_exp}")
    print(f"Maximum voltage for Simplified Young-Lippmann fitting range: {max_voltage_for_young}")
    max_voltage_index = np.argmax(voltage_data)
    max_voltage = voltage_data[max_voltage_index]
    increasing_part_voltage_data = voltage_data[:max_voltage_index + 1]
    increasing_part_angle_data = angle_data[:max_voltage_index + 1]
    decreasing_part_voltage_data = voltage_data[max_voltage_index:]
    decreasing_part_angle_data = angle_data[max_voltage_index:]

    # Plot the experimental data
    plt.plot(increasing_part_voltage_data, increasing_part_angle_data, label="Experiment - Increasing Part", marker="o", markersize = 15, linestyle="--", linewidth = 4)
    plt.plot(decreasing_part_voltage_data, decreasing_part_angle_data, label="Experiment - Decreasing Part", marker="^", markersize = 15, linestyle="--", linewidth = 4)

    # 實施 Exponential Decay Fitting
    # 布爾索引過濾
    decay_mask = (increasing_part_voltage_data >= min_voltage_for_exp) & (increasing_part_voltage_data <= max_voltage)
    filtered_voltage_data_exp = increasing_part_voltage_data[decay_mask]
    filtered_angle_data_exp = increasing_part_angle_data[decay_mask]
    print(filtered_voltage_data_exp)
    # 確認過濾後的數據是否有效
    if len(filtered_voltage_data_exp) < 3:
        print("Not enough data points for Exponential Decay fitting.")
        print(f"Data Amount: {len(filtered_voltage_data_exp)}")
        return

    # Fit the data with Exponential Decay model
    initial_guess_exp = [max(filtered_angle_data_exp), 0.01, min(filtered_angle_data_exp)]
    bounds_exp = ([max(max(filtered_angle_data_exp) - 5, 0), -np.inf, min(filtered_angle_data_exp) - 5], [max(filtered_angle_data_exp) + 5, np.inf, min(filtered_angle_data_exp) + 5])
    popt_exp_decay, _ = curve_fit(exponential_model, filtered_voltage_data_exp, filtered_angle_data_exp, p0=initial_guess_exp, bounds=bounds_exp)
    print(f"Exponential Decay POPT: {popt_exp_decay}")
    print(f"Exponential Decay theta: {exponential_model(min_voltage_for_exp, *popt_exp_decay)}")
    voltage_fit_exp = np.linspace(max(min_voltage_for_exp, 0), max_voltage, 1000)
    angle_fit_exp = exponential_model(voltage_fit_exp, *popt_exp_decay)
    
    # 計算 R^2 和 MSE
    angle_pred_exp = exponential_model(filtered_voltage_data_exp, *popt_exp_decay)
    exp_decay_r2 = r2_score(filtered_angle_data_exp, angle_pred_exp)
    exp_decay_mse = mean_squared_error(filtered_angle_data_exp, angle_pred_exp)
    print(f"Exponential Decay R^2: {exp_decay_r2}")
    print(f"Exponential Decay MSE: {exp_decay_mse}")

    # 繪製 Exponential Decay 擬合曲線
    plt.plot(voltage_fit_exp, angle_fit_exp, label='Fit - Exponential Decay Formula', linestyle='-', linewidth = 4)
    # Add R^2 and MSE annotations
    plt.annotate(f'Exp Decay $R^2$: {exp_decay_r2:.4f}\nExp Decay MSE: {exp_decay_mse:.4f}', xy=(0.6, 0.56), xycoords='axes fraction',
                 fontsize=24, horizontalalignment='left', verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

    # 實施 Young-Lippmann Fitting
    young_lippmann_mask = (increasing_part_voltage_data >= 0) & (increasing_part_voltage_data <= max_voltage_for_young)
    filtered_voltage_data_young = increasing_part_voltage_data[young_lippmann_mask]
    filtered_angle_data_young = increasing_part_angle_data[young_lippmann_mask]

    # 確保角度數據在合理範圍內
    if len(filtered_angle_data_young) < 2:
        print("Not enough data points for Young-Lippmann fitting.")
        print(f"Data Amount: {len(filtered_angle_data_young)}")
        return

    # Fit the data
    initial_guess_young = [max(filtered_angle_data_young) * np.pi / 180, 1e-6]
    bounds_young = ([(max(filtered_angle_data_young) - 0.1) * np.pi / 180, -np.inf], [(max(filtered_angle_data_young) + 0.1) * np.pi / 180, np.inf])
    popt_young_lippmann, _ = curve_fit(simplified_young_lippmann_model, filtered_voltage_data_young, filtered_angle_data_young, p0=initial_guess_young, bounds=bounds_young)
    print(f"Simplified Young Lippmann POPT: {popt_young_lippmann}")
    print(f"Simplified Young theta_0: {popt_young_lippmann[0] * 180 / np.pi}")
    voltage_fit_young = np.linspace(0, max_voltage_for_young, 1000)
    angle_fit_young = simplified_young_lippmann_model(voltage_fit_young, *popt_young_lippmann)

    # 計算 R^2 和 MSE
    angle_pred_young = simplified_young_lippmann_model(filtered_voltage_data_young, *popt_young_lippmann)
    simplified_young_lippmann_r2 = r2_score(filtered_angle_data_young, angle_pred_young)
    simplified_young_lippmann_mse = mean_squared_error(filtered_angle_data_young, angle_pred_young)
    print(f"Simplified Young-Lippmann R^2: {simplified_young_lippmann_r2}")
    print(f"Simplified Young-Lippmann MSE: {simplified_young_lippmann_mse}")

    # 繪製 Simplified Young-Lippmann 擬合曲線
    plt.plot(voltage_fit_young, angle_fit_young, label='Fit - Simplified Young-Lippmann Model', linestyle='-', linewidth = 4)
    # Add R^2 and MSE annotations
    plt.annotate(f'Young-Lippmann $R^2$: {simplified_young_lippmann_r2:.4f}\nYoung-Lippmann MSE: {simplified_young_lippmann_mse:.4f}', xy=(0.51, 0.38), xycoords='axes fraction',
                 fontsize=24, horizontalalignment='left', verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

    # 添加飽和點的計算
    threshold_slope = -0.01  # 定義斜率閾值來檢測飽和點
    slopes = np.gradient(angle_fit_exp, voltage_fit_exp)
    
    # 平滑斜率曲線
    # smoothed_slopes = np.convolve(slopes, np.ones(5)/5, mode='valid')
    
    # 找到斜率首次低於閾值的電壓
    saturation_index = np.where(slopes > threshold_slope)[0][0] if any(slopes < threshold_slope) else None
    if saturation_index is not None:
        saturation_voltage = voltage_fit_exp[saturation_index]
        plt.axvline(x=saturation_voltage, color='r', linestyle='--', linewidth = 3, label='Saturation Voltage')
        print(f"Saturation Voltage: {saturation_voltage} V")


    # 添加標籤和圖例
    plt.xlabel('Voltage (V)')
    plt.ylabel('Angle (degree)')
    plt.title(title)
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    plt.show()


title = input("請輸入標題: ")
main(title)
