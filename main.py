from src.render import nebulabrot, GPU
from src.shared import Bounds, ImageConfig, BitDepth, DynamicRangeBoost
from PIL import Image, ImageEnhance



gpu = GPU()

bounds = Bounds(-2, 2, -2, 2)
image_config = ImageConfig(
    width=1000,
    height=1000,
    bit_depth=BitDepth.EIGHT,
    dynamic_range_boost=DynamicRangeBoost.sqrt
)
equation = "add(multiply(z,z),c)"


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

image_r = ImageEnhance.Brightness(low).enhance(0.10)
image_g = ImageEnhance.Brightness(Image.blend(mid, high, 0.67)).enhance(0.85)
image_b = Image.blend(low, mid, 0.25)

rgb_image = Image.merge('RGB', (image_r, image_g, image_b))

rgb_image = ImageEnhance.Contrast(rgb_image).enhance(2)
rgb_image = rgb_image.transpose(Image.Transpose.ROTATE_270)

rgb_image.save("render.png")