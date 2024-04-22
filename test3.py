import tkinter as tk
from tkinter import filedialog, IntVar, DoubleVar, Label, Scale, Button, Frame
from PIL import Image, ImageTk
import cv2
from glob import glob
import os

class ImageApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Binary Converter")
        self.master.geometry("1200x600")

        # Initialize layout frames
        self.control_frame = Frame(self.master)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.image_frame = Frame(self.master)
        self.image_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.processed_image_frame = Frame(self.master)
        self.processed_image_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Read all the images in a directory
        self.folder_path = filedialog.askdirectory()
        if not self.folder_path:
            self.master.destroy()
            raise Exception("No folder selected, application will close.")

        self.image_files = glob(os.path.join(self.folder_path, '*.jpg')) + glob(os.path.join(self.folder_path, '*.png'))
        self.image_files.sort()
        self.current_image_index = 0
        self.load_image()

        # Image display
        self.max_width = 600  # 设置图像的最大宽度
        self.max_height = 400  # 设置图像的最大高度

        ## First Image
        self.image_label = Label(self.image_frame)
        self.image_label.pack(expand=True, fill=tk.BOTH)

        ## Second Image
        self.processed_image_label = Label(self.processed_image_frame)
        self.processed_image_label.pack(expand=True, fill=tk.BOTH)

        # Controls for image processing
        Label(self.control_frame, text="Threshold:").pack()
        self.gray_threshold = IntVar(value=0)
        self.threshold_scale = Scale(self.control_frame, from_=0, to_=255, variable=self.gray_threshold, orient='horizontal', command=self.update_image, length=300)
        self.threshold_scale.pack()

        # Gaussian blur controls
        Label(self.control_frame, text="Gaussian Blur Kernel Size:").pack()
        self.gaussian_blur_ksize = IntVar(value=3)
        self.gaussian_blur_sigmaX = IntVar(value=0)
        Scale(self.control_frame, from_=1, to_=21, variable=self.gaussian_blur_ksize, orient='horizontal', command=self.update_image, length=300).pack()

        # Contour Area
        Label(self.control_frame, text="Min Contour Area:").pack()
        self.min_contour_area = IntVar(value=0)
        Scale(self.control_frame, from_=0, to_=(self.image_height * self.image_width), variable=self.min_contour_area, orient='horizontal', command=self.update_image, length=300).pack()

        Label(self.control_frame, text="Max Contour Area:").pack()
        self.max_contour_area = IntVar(value=(self.image_height * self.image_width))
        Scale(self.control_frame, from_=1, to_=(self.image_height * self.image_width), variable=self.max_contour_area, orient='horizontal', command=self.update_image, length=300).pack()

        ## Contour Filtering
        self.contour_area = DoubleVar()

        # Save and Next button
        self.save_button = Button(self.control_frame, text="Save and Next Image", command=self.save_image)
        self.save_button.pack()

        # Update image initially
        self.update_image(None)

    def load_image(self):
        if self.current_image_index < len(self.image_files):
            image_path = self.image_files[self.current_image_index]
            self.original_image = cv2.imread(image_path)
            self.image_height, self.image_width, image_channels = self.original_image.shape
            self.filename = os.path.basename(image_path)
        else:
            self.master.destroy()
            print("No more images to process.")
            return

    # Updating Image
    def update_image(self, val):
        original_image = self.original_image.copy()
        gray_image = self.convert_to_gray(self.original_image.copy())
        blurred_image = self.apply_gaussian_blur(gray_image)
        binary_image = self.apply_gray_threshold(blurred_image)
        self.binary_image = binary_image
        edge_detection_image = self.apply_canny_edge_detection(binary_image)
        contours = self.find_and_filter_contours(edge_detection_image)
        self.draw_contours(original_image, contours)
        self.display_image(original_image, self.image_label)
        self.display_image(binary_image, self.processed_image_label)
        
    
    ## Convert to Gray
    def convert_to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## Gaussian Blur
    def apply_gaussian_blur(self, img):
        return cv2.GaussianBlur(img, (self.gaussian_blur_ksize.get(), self.gaussian_blur_ksize.get()), self.gaussian_blur_sigmaX.get())

    ## Apply Gray Threshold
    def apply_gray_threshold(self, img):
        ret, binary_im = cv2.threshold(img, self.gray_threshold.get(), 255, cv2.THRESH_BINARY)
        return binary_im
    
    ## Apply Canny Edge Detection
    def apply_canny_edge_detection(self, img):
        return cv2.Canny(img, 50, 150)

    ## Find Contours
    def find_and_filter_contours(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_filted = [c for c in contours if cv2.contourArea(c) <= self.max_contour_area.get()]
        min_filted = [c for c in max_filted if cv2.contourArea(c) >= self.min_contour_area.get()]
        return min_filted

    ## Draw Contours
    def draw_contours(self, img, contours):
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    def display_image(self, img, label):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_pil = self.resize_image(img_pil, self.max_width, self.max_height)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        label.imgtk = img_tk
        label.configure(image=img_tk)

    def resize_image(self, img_pil, max_width, max_height):
        original_width, original_height = img_pil.size
        ratio = min(max_width / original_width, max_height / original_height)
        new_size = (int(original_width * ratio), int(original_height * ratio))
        return img_pil.resize(new_size, Image.Resampling.LANCZOS)

    def save_image(self):
        save_path = os.path.join(self.folder_path, f"{os.path.splitext(self.filename)[0]}_binary.png")
        cv2.imwrite(save_path, self.binary_image)
        print(f"Image saved as {save_path}")
        self.current_image_index += 1
        self.load_image()
        self.update_image(self.threshold_scale.get())

def main():
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()

# Let's comment the main function call to avoid execution here
if __name__ == '__main__':
    main()

