import tkinter as tk
from tkinter import filedialog
import pickle

def read_and_save_voltages():
    # Set up the root window for the file dialog (will not actually show the window)
    root = tk.Tk()
    root.withdraw()

    # Open the file dialog to select a txt file
    file_path = filedialog.askopenfilename(title="Select a voltage file", filetypes=[("Text files", "*.txt")])
    if not file_path:
        print("No file selected.")
        return

    voltages = []
    # Read the voltage values from the selected file
    with open(file_path, 'r') as file:
        for line in file:
            cleaned_line = line.strip()
            if cleaned_line.endswith('v') or cleaned_line.endswith('V'):
                # Remove the last character (either 'v' or 'V') and strip any whitespace, then convert to integer
                voltages.append(int(cleaned_line[:-1].strip()))
            else:
                # Directly convert the cleaned line to integer
                voltages.append(int(cleaned_line))
        print(voltages)

    # Convert voltage strings to integers
    voltage_ints = list(map(int, voltages))

    # Prepare the filename for the .pkl file
    pkl_filename = file_path.rsplit('.', 1)[0] + '.pkl'

    # Save the voltage values into a .pkl file
    with open(pkl_filename, 'wb') as pkl_file:
        pickle.dump(voltage_ints, pkl_file)

    print(f"Voltages saved successfully to {pkl_filename}")

# Run the function
read_and_save_voltages()
