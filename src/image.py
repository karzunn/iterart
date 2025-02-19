import numpy as np
import cv2



class Image:

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.data = np.zeros(width * height, dtype=np.uint32)

    def _reshape_image(self, data: np.ndarray) -> np.ndarray:
        return data.reshape((self.height, self.width))

    def as_2d_array(self) -> np.ndarray:
        return self._reshape_image(self.data)

    def save(self, filename: str):
        log_image = np.log1p(self.data)
        max_val = np.max(log_image)
        normalized = (log_image / max_val * 65535).astype(np.uint16)
        reshaped = self._reshape_image(normalized)
        cv2.imwrite(filename, reshaped)