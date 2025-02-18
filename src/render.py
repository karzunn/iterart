import numpy as np
import pyopencl as cl
from .kernel import kernel
from .shared import Bounds
from .image import Image



class Render:

    def __init__(self, image: Image, step_size: float, max_iter: int, bounds: Bounds, bail_mag: float = 4.0):
        self.image = image
        self.step_size = step_size
        self.max_iter = max_iter
        self.bounds = bounds
        self.bail_mag = bail_mag
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
        self.c_real, self.c_imag, self.z_real, self.z_imag = self._init_arrays()

    def _init_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        real_vals = np.arange(self.bounds.x_min, self.bounds.x_max, self.step_size, dtype=np.float32)
        imag_vals = np.arange(self.bounds.y_min, self.bounds.y_max, self.step_size, dtype=np.float32)
        c_real, c_imag = np.meshgrid(real_vals, imag_vals)
        c_real, c_imag = c_real.flatten(), c_imag.flatten()
        z_real = np.zeros(len(c_real), dtype=np.uint32)
        z_imag = np.zeros(len(c_real), dtype=np.uint32)
        return c_real, c_imag, z_real, z_imag
    
    def _get_buffers(self) -> tuple[cl.Buffer, cl.Buffer, cl.Buffer, cl.Buffer, cl.Buffer]:
        d_c_real = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.c_real)
        d_c_imag = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.c_imag)
        d_z_real = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.z_real)
        d_z_imag = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.z_imag)
        d_image_data = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.image.data)
        return d_c_real, d_c_imag, d_z_real, d_z_imag, d_image_data
    
    def _collect_data(self, d_z_real: cl.Buffer, d_z_imag: cl.Buffer, d_image_data: cl.Buffer):
        cl.enqueue_copy(self.queue, self.image.data, d_image_data).wait()
        cl.enqueue_copy(self.queue, self.z_real, d_z_real).wait()
        cl.enqueue_copy(self.queue, self.z_imag, d_z_imag).wait()

    def run(self, equation: str):
        iter_count = 0
        while iter_count < self.max_iter:
            d_c_real, d_c_imag, d_z_real, d_z_imag, d_image_data = self._get_buffers()
            current_iter = min(10000, self.max_iter - iter_count)
            kernel_str = kernel(self.image, equation, current_iter, self.bail_mag, self.bounds)
            program = cl.Program(self.ctx, kernel_str).build()
            program.render(
                self.queue,
                (len(self.c_real),),
                None,
                d_c_real,
                d_c_imag,
                d_z_real,
                d_z_imag,
                d_image_data
            )
            self._collect_data(d_z_real, d_z_imag, d_image_data)

            valid = self.z_real**2 + self.z_imag**2 < self.bail_mag
            self.c_real = self.c_real[valid]
            self.c_imag = self.c_imag[valid]
            self.z_real = self.z_real[valid]
            self.z_imag = self.z_imag[valid]

            if len(self.c_real) == 0:
                break

            iter_count += current_iter