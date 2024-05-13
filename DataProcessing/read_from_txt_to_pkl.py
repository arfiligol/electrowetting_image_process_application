import os
import pickle
import tkinter as tk
from tkinter import filedialog

# Set up the root window for the file dialog
root = tk.Tk()
root.withdraw()  # Hide the root window

file_path = filedialog.askopenfilename(title="Select a angle txt file", filetypes=[("Text files", "*.txt")])
if file_path:
    data = []
    with open(file_path, "r") as txt_file:
        for line in txt_file:
            angles = [angle for angle in line.split()]
            print(angles)
            data.append(angles)
    file_dir = os.path.dirname(file_path)
    file_name, file_extension = os.path.splitext(os.path.basename(file_path))
    new_file_path = os.path.join(file_dir, file_name + ".pkl")
    with open(new_file_path, "wb") as pkl_file:
        pickle.dump(data, pkl_file)
