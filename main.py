import os
from glob import glob
import cv2
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

from image_processor import ImageProcessor
from frames import BaseFrame, ScrollableFrame, OriginImageFrame, GaussianBlurFrame, HSVFrame, HSVThresholdFrame, EdgeDetectionFrame, GrayThresholdFrame

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

        for i in range(2):
            self.window.columnconfigure(i, weight = 1)
        for i in range(3):
            self.window.rowconfigure(i, weight = 1)

        # 創建一個可滾動的 Frame 來包裹其他 Frame
        self.scroll_frame = ScrollableFrame(self.window)
        self.scroll_frame.pack(fill="both", expand=True)

        # 創建每個處理步驟的 Frame
        self.originFrame: OriginImageFrame = OriginImageFrame(self.scroll_frame.scrollable_frame, self.processor)
        self.originFrame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.gaussianFrame: GaussianBlurFrame = GaussianBlurFrame(self.scroll_frame.scrollable_frame, self.processor)
        self.gaussianFrame.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = "nsew")

        self.grayFrame: GrayThresholdFrame = GrayThresholdFrame(self.scroll_frame.scrollable_frame, self.processor)
        self.grayFrame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # self.hsvFrame = HSVFrame(self.scroll_frame.scrollable_frame, self.processor)
        # self.hsvFrame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # self.hsvThresholdFrame: HSVThresholdFrame = HSVThresholdFrame(self.scroll_frame.scrollable_frame, self.processor)
        # self.hsvThresholdFrame.grid(row = 1, column = 1, padx=10, pady=10, sticky="nsew")

        self.cannyEdgeDetectionFrame: EdgeDetectionFrame = EdgeDetectionFrame(self.scroll_frame.scrollable_frame, self.processor)
        self.cannyEdgeDetectionFrame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        # 註冊 Frame 到 BaseFrame
        BaseFrame.register_observer(self.originFrame)
        BaseFrame.register_observer(self.gaussianFrame)
        BaseFrame.register_observer(self.grayFrame)
        # BaseFrame.register_observer(self.hsvFrame)
        # BaseFrame.register_observer(self.hsvThresholdFrame)
        BaseFrame.register_observer(self.cannyEdgeDetectionFrame)

        # 初次呼叫讓圖片初始化
        BaseFrame.notify_observers()

def main():
    window = ttkb.Window(themename="litera")

    # Create ImageProcessor Instance
    image = cv2.imread("test_image4.jpg")
    processor = ImageProcessor(image)
    processor.process_image()

    # Create ImageProcessingApp Instance
    app = ImageProcessingApp(window, processor)

    window.mainloop()


if __name__ == "__main__":
    main()