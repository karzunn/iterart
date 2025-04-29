import numpy as np
import pyopencl as cl
from PIL import Image
from .kernel import kernel
from ..shared import Bounds, ImageConfig, GPU


def _init_arrays(step_size: float, bounds: Bounds) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    real_vals = np.arange(bounds.x_min, bounds.x_max, step_size, dtype=np.float32)
    imag_vals = np.arange(bounds.y_min, bounds.y_max, step_size, dtype=np.float32)
    c_real, c_imag = np.meshgrid(real_vals, imag_vals)
    c_real, c_imag = c_real.flatten(), c_imag.flatten()
    z_real = np.zeros(len(c_real), dtype=np.float32)
    z_imag = np.zeros(len(c_real), dtype=np.float32)
    return c_real, c_imag, z_real, z_imag


def nebulabrot(
        gpu: GPU,
        image_config: ImageConfig,
        equation: str,
        step_size: float,
        max_iter: int,
        bounds: Bounds,
        bail_mag: float = 4.0
    ) -> Image:
    image_data = np.zeros(image_config.height * image_config.width, dtype=np.uint32)
    c_real, c_imag, z_real, z_imag = _init_arrays(step_size, bounds)
    iter_count = 0
    while iter_count < max_iter:
        current_iter = min(10000, max_iter - iter_count)
        d_c_real = gpu.get_array_buffer(c_real, read_only=True)
        d_c_imag = gpu.get_array_buffer(c_imag, read_only=True)
        d_z_real = gpu.get_array_buffer(z_real)
        d_z_imag = gpu.get_array_buffer(z_imag)
        d_image_data = gpu.get_array_buffer(image_data)
        kernel_str = kernel(image_config, equation, current_iter, bail_mag, bounds)
        program = cl.Program(gpu.ctx, kernel_str).build()
        program.render(gpu.queue, (len(c_real),), None, d_c_real, d_c_imag, d_z_real, d_z_imag, d_image_data)
        gpu.collect_array(d_z_real, z_real)
        gpu.collect_array(d_z_imag, z_imag)
        gpu.collect_array(d_image_data, image_data)

        valid = z_real**2 + z_imag**2 < bail_mag
        c_real = c_real[valid]
        c_imag = c_imag[valid]
        z_real = z_real[valid]
        z_imag = z_imag[valid]

        if len(c_real) == 0:
            break

        iter_count += current_iter

    image_data = image_config.dr_func(image_data)
    image_data = ((image_data / np.max(image_data)) * image_config.max_val).astype(image_config.numpy_dtype)
    image_data = image_data.reshape(image_config.width, image_config.height)
    return Image.fromarray(image_data, mode=image_config.pil_mode)