from typing import Annotated

import jax
import jax.numpy as jnp
from jax import Array


from scenerep.models import RadianceField
from scenerep.renderer.camera import Camera
from scenerep.renderer.sampler import Sampler


class Renderer:
    radiance_field: RadianceField
    sampler: Sampler
    background = jnp.ones((3,))

    def __init__(self, radiance_field: RadianceField, sampler):
        self.radiance_field = radiance_field
        self.sampler = sampler

    def raycast(
        self,
        origin: Annotated[Array, ("B", 3)],
        direction: Annotated[Array, ("B", 3)],
    ):
        """Raycast a batch of rays."""

        (steps, samples) = self.sampler(origin, direction)
        steps: Annotated[Array, ("B", "S")]
        samples: Annotated[Array, ("B", "S", 3)]

        direction_tiled = direction[:, jnp.newaxis, :].repeat(steps.shape[1], 1)
        with jax.profiler.TraceAnnotation("radiance_field"):
            color, density = jax.vmap(self.radiance_field)(samples, direction_tiled)
        density = density.squeeze()

        color: Annotated[Array, ("B", "S", 3)]
        density: Annotated[Array, ("B", "S")]

        d_steps = jnp.diff(steps, prepend=0)
        d_steps: Annotated[Array, ("B", "S")]

        alpha = 1 - jnp.exp(-density * d_steps)
        alpha: Annotated[Array, ("B", "S")]
        transmittance = jnp.exp(-jnp.cumsum(density * d_steps, axis=-1))
        transmittance: Annotated[Array, ("B", "S")]

        weights = transmittance * alpha
        weights: Annotated[Array, ("B", "S")]

        integrated = jnp.einsum("BSC,BS->BC", color, weights)
        integrated: Annotated[Array, ("B", "S")]
        with_background = integrated + jnp.outer(
            1 - jnp.sum(weights, axis=1), self.background
        )

        return with_background

    def __call__(self, camera) -> jax.Array:
        """Render an entire scene."""
        origin, direction = camera.rays()  # (3), (batch, 3)
        rendered = self.raycast(origin, direction)

        return rendered.reshape(camera.height, camera.width, 3)


__all__ = ["Renderer", "Camera"]
