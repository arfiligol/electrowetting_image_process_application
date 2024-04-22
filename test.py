import tkinter as tk
from tkinter import filedialog, IntVar, DoubleVar
from PIL import Image, ImageTk
import cv2
from glob import glob
import os

class ImageApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Binary Converter")

        # Read all the images in a directory
        self.folder_path = filedialog.askdirectory()
        if not self.folder_path:
            self.master.destroy()
            raise Exception("No folder selected, application will close.")

        self.image_files = glob(os.path.join(self.folder_path, '*.jpg')) + glob(os.path.join(self.folder_path, '*.png'))
        self.image_files.sort()
        self.current_image_index = 0


        # Load an image and process
        self.load_image()

        # Attributes for image process
        ## Resize
        self.max_width = 600  # 设置图像的最大宽度
        self.max_height = 400  # 设置图像的最大高度

        ## Gaussian Blur
        self.gaussian_blur_ksize = IntVar(value=3)
        self.gaussian_blur_sigmaX = IntVar(value=0)

        ## Gray to Binary
        self.gray_threshold1 = IntVar(value=0)
        self.gray_threshold2 = IntVar(value=255)

        ## Canny Edge Detection
        self.canny_threshold1 = IntVar(value=50)
        self.canny_threshold2 = IntVar(value=150)

        ## Contour Filtering
        self.contour_area = DoubleVar()
        self.min_contour_area = IntVar(value=0)
        self.max_contour_area = IntVar(value=(self.image_height * self.image_width))



        
        # Image display
        self.image_label = tk.Label(self.master)
        self.image_label.pack()

        # Threshold control
        self.threshold_scale = tk.Scale(self.master, from_=0, to_=255, orient='horizontal', command=self.update_image)
        self.threshold_scale.pack()

        # Save Button
        self.save_button = tk.Button(self.master, text="Save and Next Image", command=self.save_image)
        self.save_button.pack()

        self.update_image()

    def load_image(self):
        if self.current_image_index < len(self.image_files):
            image_path = self.image_files[self.current_image_index]
            self.original_image = cv2.imread(image_path)
            self.image_height, self.image_width, image_channels = self.original_image.shape
            self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.filename = os.path.basename(image_path)
        else:
            self.master.destroy()
            print("No more images to process.")
            return


    # Updating Image
    def update_image(self, val):
        gray_image = self.convert_to_gray(self.original_image)
        blurred_image = self.apply_gaussian_blur(gray_image)
        binary_image = self.apply_gray_threshold(blurred_image)
        edge_detection_image = self.apply_canny_edge_detection(binary_image)
        contours = self.find_and_filter_contours(edge_detection_image)
        self.draw_contours(self.original_image, contours)
        self.display_image(binary_image)
    
    ## Convert to Gray
    def convert_to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## Gaussian Blur
    def apply_gaussian_blur(self, img):
        return cv2.GaussianBlur(img, (self.gaussian_blur_ksize.get(), self.gaussian_blur_ksize.get()), self.gaussian_blur_sigmaX.get())

    ## Apply Gray Threshold
    def apply_gray_threshold(self, img):
        ret, binary_im = cv2.threshold(img, self.gray_threshold1.get(), self.gray_threshold2.get(), cv2.THRESH_BINARY)
        return binary_im
    
    ## Apply Canny Edge Detection
    def apply_canny_edge_detection(self, img):
        return cv2.Canny(img, self.canny_threshold1.get(), self.canny_threshold2.get())

    ## Find Contours
    def find_and_filter_contours(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_filted = [c for c in contours if cv2.contourArea(c) <= self.max_contour_area.get()]
        min_filted = [c for c in max_filted if cv2.contourArea(c) >= self.min_contour_area.get()]
        return min_filted

    ## Draw Contours
    def draw_contours(self, img, contours):
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)





    def display_image(self, img):
        img_pil = Image.fromarray(img)
        img_pil = self.resize_image(img_pil, self.max_width, self.max_height)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.image_label.imgtk = img_tk
        self.image_label.configure(image=img_tk)

    def resize_image(self, img_pil, max_width, max_height):
        original_width, original_height = img_pil.size
        ratio = min(max_width/original_width, max_height/original_height)
        new_size = (int(original_width * ratio), int(original_height * ratio))
        return img_pil.resize(new_size, Image.Resampling.LANCZOS)

    def save_image(self):
        save_path = os.path.join(self.folder_path, f"{os.path.splitext(self.filename)[0]}_binary.png")
        _, binary_image = cv2.threshold(self.gray_image, self.threshold_scale.get(), 255, cv2.THRESH_BINARY)
        cv2.imwrite(save_path, binary_image)
        print(f"Image saved as {save_path}")

        self.current_image_index += 1
        if self.current_image_index < len(self.image_files):
            self.load_image()
            self.update_image(self.threshold_scale.get())
        else:
            print("Finished processing all images.")
            self.master.quit()

def main():
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
