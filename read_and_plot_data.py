import tkinter as tk
from tkinter import filedialog
import pickle
import matplotlib.pyplot as plt

def load_and_plot_data():
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
    contact_angles = [-angle for angle in contact_angles]

    # Plot the data using Matplotlib
    plt.figure()
    plt.plot(voltages, contact_angles, color='b', label='Contact Angle vs. Voltage')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Contact Angle (degrees)')
    plt.title('Contact Angle as a Function of Voltage')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function to execute the file selection and plotting
load_and_plot_data()
