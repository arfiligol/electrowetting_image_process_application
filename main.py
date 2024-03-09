import cv2
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

from image_processor import ImageProcessor
from frames import OriginImageFrame, GaussianBlurFrame, HSVFrame, HSVThresholdFrame, CannyEdgeDetectionFrame

class ImageProcessingApp:
    """ 主應用程序 App """
    def __init__(self, window: ttkb.Window, processor: ImageProcessor):
        self.window = window
        self.processor = processor
        self.init_ui()

    def init_ui(self):
        # 設置窗口布局
        self.window.title("Electrowetting Image Processing")
        self.window.geometry("1280x920")

        for i in range(3):
            self.window.columnconfigure(i, weight = 1)
        for i in range(2):
            self.window.rowconfigure(i, weight = 1)

        # 創建每個處理步驟的 Frame
        self.originFrame: OriginImageFrame = OriginImageFrame(self.window, self.processor)
        self.originFrame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.gaussianFrame: GaussianBlurFrame = GaussianBlurFrame(self.window, self.processor)
        self.gaussianFrame.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = "nsew")

        self.hsvFrame = HSVFrame(self.window, self.processor)
        self.hsvFrame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

        self.hsvThresholdFrame: HSVThresholdFrame = HSVThresholdFrame(self.window, self.processor)
        self.hsvThresholdFrame.grid(row = 1, column = 0, padx=10, pady=10, sticky="nsew")

        self.cannyEdgeDetectionFrame: CannyEdgeDetectionFrame = CannyEdgeDetectionFrame(self.window, self.processor)
        self.cannyEdgeDetectionFrame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

def main():
    window = ttkb.Window(themename="litera")

    # Create ImageProcessor Instance
    image = cv2.imread("test_image.jpg")
    processor = ImageProcessor(image)
    processor.process_image()

    # Create ImageProcessingApp Instance
    app = ImageProcessingApp(window, processor)

    window.mainloop()


if __name__ == "__main__":
    main()