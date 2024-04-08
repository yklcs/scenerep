import jax.numpy as jnp

from scenerep.models import ExampleRadianceField
from scenerep.renderer import Camera, Renderer
from scenerep.renderer.sampler import UniformSampler
from scenerep.utils import save_image

if __name__ == "__main__":
    c2w = jnp.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [5, 0, 0, 1]])
    width, height = 200, 200
    camera = Camera(c2w, width, height, jnp.sqrt(width * height))

    radiance_field = ExampleRadianceField()

    sampler = UniformSampler(0.005, 0, 5)
    renderer = Renderer(radiance_field, sampler)
    image = renderer(camera)

    save_image("out/image.png", image)
