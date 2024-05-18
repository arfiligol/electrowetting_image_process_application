import os
import re
import pickle
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

def calculate_R(data):
    """
    Calculate R Value
    data: angle v.s voltage for each concentration data
    """

    # Get \theta_0 and \theta_{end}
    theta_0 = data[0][1] # First tuple in the list, and the second element of the tuple
    theta_end = data[-1][1]

    # Get \theta_{max}
    theta_max = max(data, key=lambda x: x[0])[1] 
    # lambda accept a parameter "x" (which will be the tuple in this case), 
    # and return the first element in the tuple (which is the voltage in this case).
    # key is used to determine which thing to be compare.

    # Calculate R
    R = (theta_end - theta_max) / (theta_0 - theta_max)
    
    return R

def extract_percentage(filename):
    match = re.search(r'(\d+)%', filename)
    if match:
        return match.group(1) + '%'
    return filename


def main(title):
    # 隱藏 tkinter window
    Tk().withdraw()

    # 使用 filedialog to request a directory
    folder_path = filedialog.askdirectory(title="選擇最終檔案的資料夾")

    # 確保用戶選擇了一個資料夾
    if folder_path:
        # Get all the file paths in the directory
        file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pkl')]
        
        # Sort the file with filename
        file_paths.sort()

        # Initialize a dict to store data
        data_dict = {}

        # Iterate the file paths and store it to the dict
        for file_path in file_paths:
            with open(file_path, "rb") as file:
                data = pickle.load(file)

                # Use the filename to be the key of dict
                file_name = os.path.basename(file_path)
                data_dict[file_name] = data
        
        # Print all the key and value to make sure the data is ready
        for key, value in data_dict.items():
            print(f"Key: {key}, Data: {value}")

    else:
        print("Quit: User didn't choose any directory.")
        return
    

    # Start Processing the Data
    percentages = []
    Rs = []
    for key, value in data_dict.items():
        R = calculate_R(value)
        print(f"Key: {extract_percentage(key)}, R: {R}")
        
        # Use the percentage of concentration to be x axis
        percentages.append(extract_percentage(key))
        Rs.append(R)


    # Plot the diagram
    # Set rcParams (need to be set before plot since they are global variables)
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 12

    # Figure
    plt.figure(figsize=(12, 8))
    
    plt.plot(percentages, Rs, marker = "o", linestyle = "--")

    # Attribute of the figure
    plt.title(title)
    plt.xlabel("Weight Percentage")
    plt.ylabel("R Value")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    

title = input("Please enter a title of the figure: ")
main(title)