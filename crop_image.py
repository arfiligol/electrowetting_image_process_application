import os
import cv2
import tkinter as tk
from tkinter import Frame, Scale, Button, filedialog
from PIL import Image, ImageTk

class ImageCropperApp(tk.Tk):
    def __init__(self, window_title):
        super().__init__()
        self.title(window_title)
        self.geometry("1280x720")
        self.frames = {} # 儲存註冊的 Frame 和其排序編號
        self.image_directory = filedialog.askdirectory(title="Select Image Folder")
        self.image_files = [file for file in os.listdir(self.image_directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

        
    def register_frame(self, frame_name, frame_class, order, **kwargs):
        """ 註冊一個 Frame，並根據提供的順序編號進行排序顯示 """
        if frame_name not in self.frames:
            frame:Frame = frame_class(self, **kwargs)
            frame.pack_forget() # 初始化時不顯示
            self.frames[frame_name] = (frame, order)
            self.update_display() # 更新顯示排序
        else:
            print(f"Frame {frame_name} is already registered.")
    
    def update_display(self):
        """ 根據註冊的 Frame 的排序編號更新顯示 """
        for frame_name, (frame, order) in sorted(self.frames.items(), key=lambda x: x[1][1]):
            frame.pack()


class ResizeImageFrame(Frame):
    def __init__(self, master=None, image_path=None, **kwargs):
        super().__init__(master, **kwargs)
        self.config(bg="black", bd=2, relief=tk.SUNKEN, width=300, height=300)
        self.image_files = self.master.image_files
        self.image_files.sort()
        self.current_image_index = 0
        self.image = None # PIL ImageTk object
        self.resized_image = None # PIL ImageTk object for resized image
        self.cropped_image = None # PIL ImageTk object for cropped image

        # Create a 
        self.load_next_image()
        self.load_image()
        self.create_widgets()
        self.create_sliders()
        self.create_save_button()
        self.update_crop(None)

    def load_next_image(self):
        print("Loading next image...")
        if self.current_image_index < len(self.image_files):
            print(f"Directory path: {self.master.image_directory}")
            self.image_path = os.path.join(self.master.image_directory, self.image_files[self.current_image_index])
            print(f"File path: {self.image_path}")
            self.load_image()
        else:
            self.master.destroy()

    def load_image(self):
        """ Load an image using OpenCV and convert it for display """
        if self.image_path:

            cv_image = cv2.imread(self.image_path)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.original_image = Image.fromarray(cv_image)
            original_image_width, original_image_height = self.original_image.size
            self.original_image_width = original_image_width
            self.original_image_height = original_image_height
            self.resized_pil_image = self.original_image.resize((250, 250), Image.LANCZOS)
            self.original_pil_image = ImageTk.PhotoImage(self.original_image)
            self.resized_image = ImageTk.PhotoImage(self.resized_pil_image)
            self.cropped_image = ImageTk.PhotoImage(Image.new('RGB', (250, 250), "black"))

    def create_widgets(self):
        """ Create GUI widgets """
        # if self.resized_image:
        #     self.original_image_label = tk.Label(self, image=self.resized_image)
        # else:
        #     self.original_image_label = tk.Label(self, text="Unable to load image")
        # self.original_image_label.pack()
        
        if self.cropped_image:
            self.cropped_image_label = tk.Label(self, image=self.cropped_image)
        else:
            self.cropped_image_label = tk.Label(self, text="Unable to perform image")
        self.cropped_image_label.pack()

    def create_sliders(self):
        self.width_size_slider = Scale(self, from_=50, to=(self.original_image_width / 2), length=(self.original_image_width / 2 - 50), orient="horizontal", label="Crop Width Size")
        self.width_size_slider.pack()
        self.width_size_slider.bind("<Motion>", self.update_crop)

        self.height_size_slider = Scale(self, from_=50, to=(self.original_image_height / 2), length=(self.original_image_height / 2 - 50), orient="horizontal", label="Crop Height Size")
        self.height_size_slider.pack()
        self.height_size_slider.bind("<Motion>", self.update_crop)


        self.x_slider = Scale(self, from_=0, to=self.original_image_width, length=500, orient="horizontal", label="Center X")
        self.x_slider.set(250)
        self.x_slider.pack()
        self.x_slider.bind("<Motion>", self.update_crop)

        self.y_slider = Scale(self, from_=0, to=self.original_image_height, length=500, orient="horizontal", label="Center Y")
        self.y_slider.set(250)
        self.y_slider.pack()
        self.y_slider.bind("<Motion>", self.update_crop)

    def update_crop(self, event):
        width_size = self.width_size_slider.get()
        height_size = self.height_size_slider.get()
        center_x = self.x_slider.get()
        center_y = self.y_slider.get()

        self.left = max(center_x - width_size // 2, 0)
        self.upper = max(center_y - height_size // 2, 0)
        self.right = min(center_x + width_size // 2, self.original_image_width)
        self.lowerr = min(center_y + height_size // 2, self.original_image_height)

        cropped_image = self.original_image.crop((self.left, self.upper, self.right, self.lowerr))
        # cropped_image = cropped_image.resize((250, 250), Image.LANCZOS)
        self.cropped_image = ImageTk.PhotoImage(cropped_image)
        self.cropped_image_label.config(image=self.cropped_image)

    def create_save_button(self):
        save_button = Button(self, text="Save Cropped Image", command=self.save_cropped_image)
        save_button.pack()

    def save_cropped_image(self):
        """Save the cropped image to disk with a modified filename."""
        if self.image_path and self.cropped_image:
            file_directory = os.path.dirname(self.image_path)
            new_folder_path = os.path.join(file_directory, "cut")
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            file_name, ext = os.path.splitext(os.path.basename(self.image_path))
            new_filename = file_name + " - cut" + ext
            print(f"New Path: {new_folder_path}")
            print(f"New Filename: {new_filename}")
            print(f"Extension: {ext}")
            save_path = os.path.join(new_folder_path, new_filename)
            save_pil_image = self.original_image.crop((self.left, self.upper, self.right, self.lowerr))
            save_pil_image.save(save_path)
            print(f"Saved cropped image as {save_path}")
        self.current_image_index += 1
        self.load_next_image()
        self.update_crop(None)


if __name__ == "__main__":
    app = ImageCropperApp("Image Cropper and Resizer App")
    
    image_path = "data\\3mm -1\IMG20240410181435.jpg"
    app.register_frame("resize_image_frame", ResizeImageFrame, 2, image_path=image_path)
    
    app.mainloop()