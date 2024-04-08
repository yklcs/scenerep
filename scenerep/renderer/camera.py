from jax import Array
import jax.numpy as jnp
from typing import Annotated


class Camera:
    camera_to_world: Annotated[Array, (4, 4)]
    width: int
    height: int
    f: float

    def __init__(self, camera_to_world, width, height, f):
        self.camera_to_world = camera_to_world
        self.width = width
        self.height = height
        self.f = f

    @property
    def size(self):
        return self.width * self.height

    def rays(self) -> tuple[Annotated[Array, (3,)], Annotated[Array, ("S", 3)]]:
        x, y = jnp.meshgrid(
            jnp.arange(self.width), jnp.arange(self.height)
        )  # projected plane coordinates
        cx, cy = self.width / 2, self.height / 2  # centers

        origin = self.camera_to_world[3, :3]

        dirs_cam = jnp.stack(
            ((x - cx) / self.f, (y - cy) / self.f, jnp.ones_like(x)), axis=-1
        ).reshape(-1, 3)
        dirs_world = dirs_cam @ self.camera_to_world[:3, :3].T
        dirs = dirs_world / jnp.linalg.norm(dirs_world, axis=1, keepdims=True)

        return origin, dirs
