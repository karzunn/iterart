from PIL import Image
from ..shared import ImageConfig, GPU, Bounds
import numpy as np
import pyopencl as cl


def _init_arrays(step_size: float, bounds: Bounds) -> tuple[np.ndarray, np.ndarray]:
    x_vals = np.arange(bounds.x_min, bounds.x_max, step_size, dtype=np.float32)
    y_vals = np.arange(bounds.y_min, bounds.y_max, step_size, dtype=np.float32)
    x_vals, y_vals = np.meshgrid(x_vals, y_vals)
    return x_vals.flatten(), y_vals.flatten()


def clifford(
    gpu: GPU,
    image_config: ImageConfig,
    step_size: float,
    max_iter: int,
    bounds: Bounds,
    a: float,
    b: float,
    c: float,
    d: float
) -> Image:
    
    x_vals, y_vals = _init_arrays(step_size, bounds)
    image_data = np.zeros(image_config.height * image_config.width, dtype=np.uint32)

    d_x_vals = gpu.get_array_buffer(x_vals)
    d_y_vals = gpu.get_array_buffer(y_vals)
    d_image_data = gpu.get_array_buffer(image_data)
    kernel_str = kernel(image_config, max_iter, bounds, a, b, c, d)
    program = cl.Program(gpu.ctx, kernel_str).build()
    program.render(gpu.queue, (len(x_vals),), None, d_x_vals, d_y_vals, d_image_data)
    gpu.collect_array(d_image_data, image_data)

    image_data = image_config.dr_func(image_data)
    image_data = ((image_data / np.max(image_data)) * image_config.max_val).astype(image_config.numpy_dtype)
    image_data = image_data.reshape(image_config.width, image_config.height)
    return Image.fromarray(image_data, mode=image_config.pil_mode)


def kernel(
    image_config: ImageConfig,
    max_iter: int,
    bounds: Bounds,
    a: float,
    b: float,
    c: float,
    d: float
) -> str:
    return """
__kernel void render(
    __global float *x, __global float *y, __global int *image_data
) 
{{
    int i = get_global_id(0);
    float local_x = x[i];
    float local_y = y[i];
    float new_local_x = 0.0;
    float new_local_y = 0.0;

    for (int iter = 0; iter < {max_iter}; iter++) {{

        new_local_x = sin({a}f * local_y) + {c}f * cos({a}f * local_x);
        new_local_y = sin({b}f * local_x) + {d}f * cos({b}f * local_y);
        local_x = new_local_x;
        local_y = new_local_y;

        int px = (int)((local_x - {x_min}f) / ({x_max}f - {x_min}f) * {width});
        int py = (int)((local_y - {y_min}f) / ({y_max}f - {y_min}f) * {height});
        if (px >= 0 && px < {width} && py >= 0 && py < {height}) {{
            atomic_inc(&image_data[py * {width} + px]);
        }}
    }}
}}
""".format(
    max_iter=max_iter,
    x_min=float(bounds.x_min),
    x_max=float(bounds.x_max),
    y_min=float(bounds.y_min),
    y_max=float(bounds.y_max),
    width=image_config.width,
    height=image_config.height,
    a=float(a),
    b=float(b),
    c=float(c),
    d=float(d)
)