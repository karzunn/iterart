from src.render import nebulabrot, GPU
from src.shared import Bounds, ImageConfig, BitDepth
from PIL import Image



gpu = GPU()

bounds = Bounds(-2, 2, -2, 2)
image_config = ImageConfig(
    width=1000,
    height=1000,
    bit_depth=BitDepth.EIGHT
)
equation = "add(squared(z), c)"


image_r = nebulabrot(
    gpu=gpu,
    image_config=image_config,
    equation=equation,
    step_size=0.002,
    max_iter=5000,
    bounds=bounds
)

image_g = nebulabrot(
    gpu=gpu,
    image_config=image_config,
    equation=equation,
    step_size=0.002,
    max_iter=10000,
    bounds=bounds
)

image_b = nebulabrot(
    gpu=gpu,
    image_config=image_config,
    equation=equation,
    step_size=0.003,
    max_iter=20000,
    bounds=bounds
)

rgb_image = Image.merge('RGB', (image_r, image_g, image_b))

rgb_image.show()
