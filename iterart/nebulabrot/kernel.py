from ..shared import Bounds, ImageConfig


def kernel(
        image_config: ImageConfig,
        equation: str,
        max_iter: int,
        bail_mag: float,
        bounds: Bounds
    ) -> str:
    return """
typedef struct {{
    float real;
    float imag;
}} Complex;

Complex add(Complex a, Complex b) {{
    Complex result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}}

Complex subtract(Complex a, Complex b) {{
    Complex result;
    result.real = a.real - b.real;
    result.imag = a.imag - b.imag;
    return result;
}}

Complex multiply(Complex a, Complex b) {{
    Complex result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}}

Complex divide(Complex a, Complex b) {{
    float denom = b.real * b.real + b.imag * b.imag;
    Complex result;
    result.real = (a.real * b.real + a.imag * b.imag) / denom;
    result.imag = (a.imag * b.real - a.real * b.imag) / denom;
    return result;
}}

float mag_squared(Complex z) {{
    return z.real * z.real + z.imag * z.imag;
}}


__kernel void render(
    __global float *c_real, __global float *c_imag,
    __global float *z_real, __global float *z_imag,
    __global int *image_data
) 
{{
    int i = get_global_id(0);
    Complex c = {{ c_real[i], c_imag[i] }};
    Complex z = {{ z_real[i], z_imag[i] }};
    Complex locations[{max_iter}];
    int count = 0;

    for (int iter = 0; iter < {max_iter}; iter++) {{
        {equation};
        locations[iter] = z;
        count++;
        if (mag_squared(z) > {bail_mag}f) break;
    }}

    if (mag_squared(z) > {bail_mag}f) {{
        for (int j = 0; j < count; j++) {{
            int px = (int)((locations[j].real - {x_min}f) / ({x_max}f - {x_min}f) * {width});
            int py = (int)((locations[j].imag - {y_min}f) / ({y_max}f - {y_min}f) * {height});

            if (px >= 0 && px < {width} && py >= 0 && py < {height}) {{
                atomic_inc(&image_data[py * {width} + px]);
            }}
        }}
    }} else {{
        z_real[i] = z.real;
        z_imag[i] = z.imag;
    }}
}}
""".format(
    equation=equation,
    max_iter=max_iter,
    bail_mag=float(bail_mag),
    x_min=float(bounds.x_min),
    x_max=float(bounds.x_max),
    y_min=float(bounds.y_min),
    y_max=float(bounds.y_max),
    width=image_config.width,
    height=image_config.height
)