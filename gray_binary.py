import tkinter as tk
from tkinter import filedialog, IntVar, DoubleVar, Label, Scale, Button, Frame
from PIL import Image, ImageTk
import cv2
from glob import glob
import os
import numpy as np
import pickle

class ImageApp:
    def __init__(self, master: "tk.Tk"):
        self.master = master
        self.master.title("Image Binary Converter")
        self.master.geometry("1200x600")

        # Initialize layout frames
        self.control_frame = Frame(self.master)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.Y, padx=10, pady=10)

        self.image_frame = Frame(self.master)
        self.image_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.processed_image_frame = Frame(self.master)
        self.processed_image_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Read all the images in a directory
        self.folder_path = filedialog.askdirectory()
        if not self.folder_path:
            self.master.destroy()
            raise Exception("No folder selected, application will close.")

        self.voltage_pkl_path = filedialog.askopenfilename(title="Select Voltage File (pkl for python)", filetypes=[("Pickle files", "*.pkl")])
        if not self.voltage_pkl_path:
            self.master.destroy()
            raise Exception("No voltage file is selected.")
        else:
            with open(self.voltage_pkl_path, "rb") as file:
                self.voltages = pickle.load(file)
                print(self.voltages)
        self.data_pairs = [] # 用來儲存角度對電壓的 tuple

        self.image_files = glob(os.path.join(self.folder_path, '*.jpg')) + glob(os.path.join(self.folder_path, '*.png'))
        self.image_files.sort()
        if len(self.image_files) != len(self.voltages):
            print("電壓與圖片數量 not match.")
            self.master.destroy()
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
        Scale(self.control_frame, from_=1, to_=51, variable=self.gaussian_blur_ksize, orient='horizontal', command=self.update_image, length=300).pack()

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
        Label(self.control_frame, text="Adjust Filtering Contour Vertical X1 (Smaller Value):").pack()
        self.filtering_contour_x1_value = IntVar(value=(self.image_width // 2 - self.image_width // 4))
        Scale(self.control_frame, from_=0, to_=(self.image_width // 2), variable=self.filtering_contour_x1_value, orient="horizontal", command=self.update_image, length=(self.image_width // 2)).pack()

        Label(self.control_frame, text="Adjust Filtering Contour Vertical X2 (Bigger Value): ").pack()
        self.filtering_contour_x2_value = IntVar(value=(self.image_width // 2 + self.image_width // 4))
        Scale(self.control_frame, from_=0, to_=(self.image_width // 2), variable=self.filtering_contour_x2_value, orient="horizontal", command=self.update_image, length=(self.image_width // 2)).pack()


        # Adjusting Horizontal Line
        Label(self.control_frame, text="Adjust Y1 (Smaller, Higher Value): ").pack()
        self.filtering_contour_y1_value = IntVar(value=0)
        Scale(self.control_frame, from_=-300, to_=300, variable=self.filtering_contour_y1_value, orient="horizontal", command=self.update_image, length=600).pack()

        Label(self.control_frame, text="Adjust Y2 (Bigger, Lower Value): ").pack()
        self.filtering_contour_y2_value = IntVar(value=0)
        Scale(self.control_frame, from_=-300, to_=300, variable=self.filtering_contour_y2_value, orient="horizontal", command=self.update_image, length=600).pack()


        # Error of Fitting
        self.line_fitting_mse_label = Label(self.control_frame, text="Line Fitting MSE Value: ")
        self.line_fitting_mse_label.pack()

        # Contact Angle
        self.contact_angle_label = Label(self.control_frame, text="Contact Angle: ")
        self.contact_angle_label.pack()

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
        max_filted = [c2 for c2 in contours if cv2.contourArea(c2) <= self.max_contour_area.get()]
        min_filted = [c2 for c2 in max_filted if cv2.contourArea(c2) >= self.min_contour_area.get()]
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
                
                # 找到最左邊（x 最小）的點
                min_x_point = min(cnt, key=lambda point: point[0][0])
                # 將 filtering_contour_x1_value 設為最左邊的點的 x 值
                c_x1 = min_x_point[0][0] + self.filtering_contour_x1_value.get()
                print(f"最左邊的 x 值: {min_x_point[0][0]}")
                c_x2 = min_x_point[0][0] + 20 + self.filtering_contour_x2_value.get()

                # 計算y值的直方圖
                hist, bin_edges = np.histogram(y_values, bins=range(int(min(y_values)), int(max(y_values)) + 1))

                # 找到最大頻率的y值，這將是最可能的平面邊緣的y值
                c = bin_edges[np.argmax(hist)]
                c1 = min_x_point[0][1] -20 - self.filtering_contour_y1_value.get()
                c2 = min_x_point[0][1] - self.filtering_contour_y2_value.get()
                print("垂直線 x1, x2 值: ({}, {})".format(c_x1, c_x2))
                print("水平線 c1, c2 值: ({}, {})".format(c1, c2))

                # 繪製水平線與垂直線 (用於 filter contour)
                cv2.line(img, (c_x1, 0), (c_x1, img.shape[0]), (0, 0, 255), 2)
                cv2.line(img, (c_x2, 0), (c_x2, img.shape[0]), (0, 0, 255), 2)
                cv2.line(img, (0, int(c1)), (img.shape[1], int(c1)), (0, 0, 255), 2)
                cv2.line(img, (0, int(c2)), (img.shape[1], int(c2)), (0,0,255), 2)
                # 濾掉邊界以外的點
                filtered_contour = cnt[(cnt[:, 0, 0] >= c_x1) & (cnt[:, 0, 0] <= c_x2)]
                filtered_contour = filtered_contour[(filtered_contour[:, 0, 1] >= c1) & (filtered_contour[:, 0, 1] <= c2)]

                # 確保 filtered_contour 有內容（邊界範圍正確）
                # print(f"Filtered Contour: {filtered_contour}")
                if filtered_contour.any(): # filtered_contour 是 numpy array
                    # 直線擬和
                    # 使用 fitLine 獲得直線參數
                    [vx, vy, x0, y0] = cv2.fitLine(filtered_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    print("vx: {}, vy: {}, x0: {}, y0: {}".format(vx, vy, x0, y0))

                    # 在圖片上繪製直線
                    ## 計算直線的兩個點
                    t1 = 1000  # 可以調整這個值以適應你的圖像大小
                    x1 = int(x0 + t1 * vx)
                    y1 = int(y0 + t1 * vy)
                    x2 = int(x0 - t1 * vx)
                    y2 = int(y0 - t1 * vy)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 藍色直線，線寬為 2

                    # 計算斜率
                    m = vy / vx
                    print(f"斜率: {m}")
                    
                    # 計算直線方程 (Ax + By + C = 0)
                    A = vy
                    B = -vx
                    C = vx * y0 - vy * x0

                    # 計算 MSE
                    ## 計算每個點到直線的距離的平方 [(abs(A*point[0][0] + B*point[0][1] + C) / np.sqrt(A**2 + B**2)]
                    distances_squared = [(abs(A * point[0][0] + B * point[0][1] + C) / np.sqrt(A**2 + B**2))**2 for point in filtered_contour]
                    mse = np.mean(distances_squared)
                    self.line_fitting_mse_label.config(text="Line Fitting MSE Value: {:.4f}".format(mse))

                    # 計算接觸角
                    if isinstance(m, np.ndarray) and m.size == 1:
                        m = float(m.item())  # This extracts the single item as a float from the array

                        contact_angle = np.degrees(np.arctan(m))
                        self.current_contact_angle = contact_angle
                        print("Contact Angle: {:.5f}".format(m))
                        self.contact_angle_label.config(text="Contact Angle: {:.5f}".format(contact_angle))

        else:
            print(f"Get {len(contours)} contours.")

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
        # Ensure the 'binary' subdirectory exists
        binary_folder = os.path.join(self.folder_path, "binary")
        os.makedirs(binary_folder, exist_ok=True)

        # Construct the path to save the binary image
        base_filename = os.path.splitext(self.filename)[0]
        save_path = os.path.join(binary_folder, f"{base_filename}_binary.png")
        cv2.imwrite(save_path, self.binary_image)
        print(f"Image saved as {save_path}")

        # 儲存資料
        data_pair = (self.voltages[self.current_image_index], self.current_contact_angle)
        self.data_pairs.append(data_pair)

        self.current_image_index += 1     
        if self.current_image_index < len(self.voltages):
            self.load_image()
            self.update_image(self.threshold_scale.get())
        else:
            pkl_filename = self.voltage_pkl_path.rsplit(".", 1)[0] + "- data" + ".pkl"
            with open(pkl_filename, "wb") as data_file:
                pickle.dump(self.data_pairs, data_file)
                print("Succesfully write data to pkl file.")
            print("All the data has been sucessfully proccessed. Quit the App.")
            self.master.destroy()

def main():
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()

# Let's comment the main function call to avoid execution here
if __name__ == '__main__':
    main()

