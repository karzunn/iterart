from src.image import Image
from src.render import Render
from src.shared import Bounds

image = Image(width=5000, height=5000)
render = Render(
    image,
    step_size=0.001,
    max_iter=100000,
    bounds=Bounds(-2, 2, -2, 2)
)
render.run("add(squared(z), c)")
image.save("render.png")