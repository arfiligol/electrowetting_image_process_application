from .base_frame import BaseFrame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..image_processor import ImageProcessor

class HSVFrame(BaseFrame):
    """BGR to HSV 的 Frame"""
    def __init__(self, parent, processor: 'ImageProcessor', *args, **kwargs):
        super().__init__(parent, "HSV", *args, **kwargs)
        self.processor = processor
