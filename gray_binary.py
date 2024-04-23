import tkinter as tk
from tkinter import filedialog, IntVar, DoubleVar, Label, Scale, Button, Frame
from PIL import Image, ImageTk
import cv2
from glob import glob
import os
import numpy as np

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

        # Vertical Line For Filtering Contour
        Label(self.control_frame, text="Filtering Contour Vertical X:").pack()
        self.vertical_x_vlaue = IntVar(value=(self.image_width // 2))
        Scale(self.control_frame, from_=0, to_=self.image_width, variable=self.vertical_x_vlaue, orient="horizontal", command=self.update_image, length=self.image_width).pack()

        # Adjusting Horizontal Line
        Label(self.control_frame, text="Adjust Y Value:").pack()
        self.adjust_y_value_amount = IntVar(value=0)
        Scale(self.control_frame, from_=-300, to_=300, variable=self.adjust_y_value_amount, orient="horizontal", command=self.update_image, length=600).pack()

        # Error of Fitting
        self.ellipse_fitting_mse_label = Label(self.control_frame, text="Ellipse Fitting MSE Value: ")
        self.ellipse_fitting_mse_label.pack()

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
        self.fit_ellipse_and_draw_it(original_image, contours)
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

    ## Fit Ellipse
    def fit_ellipse_and_draw_it(self, img, contours):
        if (len(contours) == 1):
            print("Only one contour")
            for cnt in contours:

                # 獲取輪廓點的所有y值
                y_values = [point[0][1] for point in cnt]

                # 計算y值的直方圖
                hist, bin_edges = np.histogram(y_values, bins=range(int(min(y_values)), int(max(y_values)) + 1))

                # 找到最大頻率的y值，這將是最可能的平面邊緣的y值
                c = bin_edges[np.argmax(hist)]
                c += self.adjust_y_value_amount.get()
                print("垂直線 x 值: " + str(self.vertical_x_vlaue.get()))
                print("水平線 y 值: " + str(c))

                # 繪製水平線與垂直線 (用於 filter contour)
                cv2.line(img, (self.vertical_x_vlaue.get(), 0), (self.vertical_x_vlaue.get(), img.shape[0]), (255, 0, 0), 2)
                cv2.line(img, (0, int(c)), (img.shape[1], int(c)), (255,0,0), 2)

                # 濾掉水平線以下的點
                filtered_contour = cnt[cnt[:, 0, 0] < self.vertical_x_vlaue.get()]
                filtered_contour = filtered_contour[filtered_contour[:, 0, 1] < c]

                # # 多項式擬和
                # degree = 2
                # coefficients = np.polyfit(filtered_contour[:, 0][:, 0], filtered_contour[:, 0][:, 1], degree)
                # polynomial = np.poly1d(coefficients)

                # x_fit = np.linspace(min(filtered_contour[:, 0][:, 0]), max(filtered_contour[:, 0][:, 0]), 400)
                # y_fit = polynomial(x_fit)


                # 橢圓擬和
                ellipse = cv2.fitEllipse(filtered_contour)

                # Extract the center, axis lengths, and rotation angle from the ellipse parameters.
                (x0, y0), (MA, ma), angle = ellipse

                # 生成橢圓上的點
                ellipse_points = cv2.ellipse2Poly(
                    (int(x0), int(y0)),           # 中心座標 (x0, y0)
                    (int(MA) // 2, int(ma) // 2), # 長軸和短軸的長度 (MA, ma)
                    int(angle),                   # 旋轉角度 angle
                    0, 360, 1                     # 起始角度、結束角度、角度增量
                )
                
                # 計算每個輪廓點到橢圓的最近距離
                distances = [cv2.pointPolygonTest(ellipse_points, (int(pt[0][0]), int(pt[0][1])), True) for pt in filtered_contour]

                # 計算均方誤差
                mse = np.mean(np.square(distances))
                self.ellipse_fitting_mse_label.config(text="Ellipse Fitting MSE Value: {:.4f}".format(mse))
                print(f"Mean squared error of the fit: {mse}")

                # 解橢圓方程找焦點
                y_prime = (c - y0) * np.cos(angle) + (c - y0) * np.sin(angle)
                x_prime = ma**2 * (1 - y_prime**2 / ma**2)**0.5
                x = x0 + x_prime * np.cos(angle) - y_prime * np.sin(angle)

                # 計算斜率
                m = - (ma**2 * x_prime) / (MA**2 * y_prime)
                print(f"斜率: {m}")

                # 計算接觸角
                contact_angle = np.degrees(np.arctan(m)) + angle # 修正旋轉角度回去
                print(f"接觸角: {contact_angle}")

                # 繪製 fit 橢圓 and most point 到 contour image 上
                cv2.ellipse(img, ellipse, (0,0,255), 2)




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

