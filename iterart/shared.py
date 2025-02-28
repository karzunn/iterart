from enum import Enum
from typing import Callable
import numpy as np
import pyopencl as cl


class GPU:
    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags


class Bounds:

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max


class BitDepth(Enum):
    EIGHT = 8
    SIXTEEN = 16
    THIRTY_TWO = 32


class DynamicRangeBoost(Enum):
    log = "log"
    sqrt = "sqrt"


class ImageConfig:

    def __init__(self, width: int, height: int, bit_depth: BitDepth, dynamic_range_boost: DynamicRangeBoost):
        self.width = width
        self.height = height
        self.bit_depth = bit_depth
        self.dynamic_range_boost = dynamic_range_boost

    @property
    def max_val(self) -> int:
        return 2 ** self.bit_depth.value - 1
    
    @property
    def numpy_dtype(self):
        numpy_dtype_mapping = {
            BitDepth.EIGHT: np.uint8,
            BitDepth.SIXTEEN: np.uint16,
            BitDepth.THIRTY_TWO: np.uint32
        }
        return numpy_dtype_mapping[self.bit_depth]
    
    @property
    def pil_mode(self) -> str:
        pil_mode_mapping = {
            BitDepth.EIGHT: 'L',
            BitDepth.SIXTEEN: 'I;16',
            BitDepth.THIRTY_TWO: 'I'
        }
        return pil_mode_mapping[self.bit_depth]
    
    @property
    def dr_func(self) -> Callable:
        dr_func_mapping = {
            DynamicRangeBoost.log: np.log1p,
            DynamicRangeBoost.sqrt: np.sqrt
        }
        return dr_func_mapping[self.dynamic_range_boost]


def get_array_buffer(gpu: GPU, arr: np.ndarray, read_only: bool = False) -> cl.Buffer:
    if read_only:
        return cl.Buffer(gpu.ctx, gpu.mf.READ_ONLY | gpu.mf.COPY_HOST_PTR, hostbuf=arr)
    return cl.Buffer(gpu.ctx, gpu.mf.READ_WRITE | gpu.mf.COPY_HOST_PTR, hostbuf=arr)


def collect_array(gpu: GPU, buffer: cl.Buffer, arr: np.ndarray):
    cl.enqueue_copy(gpu.queue, arr, buffer).wait()