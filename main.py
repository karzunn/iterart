from src.image import Image
from src.render import nebulabrot, GPU
from src.shared import Bounds

gpu = GPU()
image = Image(width=1000, height=1000)
image = nebulabrot(
    gpu=gpu,
    image=image,
    equation="add(squared(z), c)",
    step_size=0.001,
    max_iter=10000,
    bounds=Bounds(-2, 2, -2, 2)
)
image.save("render.png")