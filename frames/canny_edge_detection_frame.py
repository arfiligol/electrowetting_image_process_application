import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

from .base_frame import BaseFrame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from image_processor import ImageProcessor


class CannyEdgeDetectionFrame(BaseFrame):
    """Canny邊緣檢測步驟的Frame"""
    def __init__(self, parent, processor: 'ImageProcessor', *args, **kwargs):
        super().__init__(parent, "Canny Edge Detection", *args, **kwargs)
        self.processor = processor

        self.lowThresholdSlider = ttkb.Scale(self, from_=0, to=100, orient='horizontal', )
        self.lowThresholdSlider.pack(fill='x', padx=10, pady=5)
        self.highThresholdSlider = ttkb.Scale(self, from_=100, to=200, orient='horizontal', )
        self.highThresholdSlider.pack(fill='x', padx=10, pady=5)

