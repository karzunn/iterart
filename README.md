# Overview
```iterart``` is a collection of tools to generate renderings based on iterative functions.

# Installation
```
pip install iterart
```

# Render Functions

## Nebulabrot

### Examples


```python
from iterart.nebulabrot import nebulabrot
from iterart.shared import Bounds, ImageConfig, BitDepth, DynamicRangeBoost, GPU

gpu = GPU() #You will be prompted to select which GPU to utilize
bounds = Bounds(-2, 2, -2, 2) #Specify the region in the complex plane to render on
image_config = ImageConfig(
    width=1000,
    height=1000,
    bit_depth=BitDepth.EIGHT,
    dynamic_range_boost=DynamicRangeBoost.sqrt #Takes the square root of the final pixel values to improve dynamic range.
)
equation = "z=add(multiply(z,z),c)" #The equation to render. Here we render z=z^2+c.

rendering = nebulabrot(
    gpu=gpu,
    image_config=image_config,
    equation=equation,
    step_size=0.001, #A grid scan is performed when choosing values of "c". This is the spacing used.
    max_iter=5000,
    bounds=bounds
)

#Renderings are PIL images, so we can use the "save" method.
rendering.save("render.png")
```
![](samples/nebulabrot.png)

Equations are written in OpenCL. Every equation has access to a "z" and "c" value. The "c" value is a location on the complex plane. The "z" value will always begin at zero for the iterations. Both values are structs of type 'Complex' and require the following functions to manipulate them:

- add
- subtract
- multiply
- divide

You can access the imaginary and real components using the 'imag' and 'real' properties. For example:
```python
equation = """
Complex neg_imag = { -z.imag, 0 };
z=add(add(multiply(z,z),c),neg_imag)
"""
```
![](samples/imaginary.png)

To produce color renderings, you can combine grayscale renderings into an RGB image. Since renderings are PIL images, we can use tools from that package to accomplish this.

```python
from iterart.nebulabrot import nebulabrot
from iterart.shared import Bounds, ImageConfig, BitDepth, DynamicRangeBoost, GPU
from PIL import Image, ImageEnhance


gpu = GPU()
bounds = Bounds(-2, 2, -2, 2)
image_config = ImageConfig(
    width=1000,
    height=1000,
    bit_depth=BitDepth.EIGHT,
    dynamic_range_boost=DynamicRangeBoost.sqrt
)
equation = """
z=add(multiply(z,z),c)
"""

# Choosing different max iterations will produce slightly different images.
low = nebulabrot(
    gpu=gpu,
    image_config=image_config,
    equation=equation,
    step_size=0.001,
    max_iter=5000,
    bounds=bounds
)

mid = nebulabrot(
    gpu=gpu,
    image_config=image_config,
    equation=equation,
    step_size=0.001,
    max_iter=10000,
    bounds=bounds
)

high = nebulabrot(
    gpu=gpu,
    image_config=image_config,
    equation=equation,
    step_size=0.0015,
    max_iter=20000,
    bounds=bounds
)

image_r = ImageEnhance.Brightness(low).enhance(0.10) #Our goal is an overall blue image, so we can reduce the red channel.
image_g = ImageEnhance.Brightness(Image.blend(mid, high, 0.67)).enhance(0.85) #Green will be made mostly from the higher iteration rendering, which will result in green accents. We also will reduce it overall to still favor a more blue hue.
image_b = Image.blend(low, mid, 0.25)

rgb_image = Image.merge('RGB', (image_r, image_g, image_b))

rgb_image = ImageEnhance.Contrast(rgb_image).enhance(2)
rgb_image = rgb_image.transpose(Image.Transpose.ROTATE_270)

rgb_image.save("color.png")
```
![](samples/color.png)

## Attractors

### Clifford

```python
from matplotlib import pyplot as plt
import numpy as np
from iterart.attractors import clifford
from iterart.shared import Bounds, ImageConfig, BitDepth, DynamicRangeBoost, GPU
from PIL import Image


gpu = GPU()

bounds = Bounds(-3, 3, -3, 3)
image_config = ImageConfig(
    width=1000,
    height=1000,
    bit_depth=BitDepth.EIGHT,
    dynamic_range_boost=DynamicRangeBoost.log
)

render = clifford(
    gpu=gpu,
    image_config=image_config,
    step_size=0.02, #A grid scan is performed when choosing initial x and y values, this is the spacing used.
    max_iter=10000,
    bounds=bounds,
    a=1.7, b=1.7, c=0.6, d=1.2 #These correspond to the constants of the clifford attractor equations.
)

# Apply colormap from matplotlib (e.g., 'inferno', 'viridis', 'plasma')
image_arr = np.array(render)
colormap = plt.get_cmap("afmhot")
rgba = colormap(image_arr)

# Convert RGBA to RGB (discard the alpha channel)
rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

# Convert back to PIL image
rgb_render = Image.fromarray(rgb)
rgb_render.save("clifford.png")
```
![](samples/clifford.png)