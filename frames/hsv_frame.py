from .base_frame import BaseFrame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..image_processor import ImageProcessor

class HSVFrame(BaseFrame):
    """BGR to HSV 的 Frame"""
    def __init__(self, parent, processor: 'ImageProcessor', *args, **kwargs):
        self.processor = processor
        super().__init__(parent, "HSV", *args, **kwargs)

    def update_image(self, img = None):
        if img is None:
            img = self.processor.hsv
        super().update_image(img)