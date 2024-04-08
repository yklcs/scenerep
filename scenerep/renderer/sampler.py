from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp


class Sampler(ABC):
    @abstractmethod
    def __call__(
        self, origin: jax.Array, direction: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Calculate steps and samples at steps"""
        ...


@dataclass
class UniformSampler(Sampler):
    step_size: float
    near: float = 0.0
    far: float = 1.0

    def __call__(self, origin: jax.Array, direction: jax.Array):
        steps = jnp.arange(self.near, self.far, self.step_size)
        steps = jnp.tile(steps, (direction.shape[0], 1))
        samples = origin + jnp.einsum("bd,bs->bsd", direction, steps)

        return (steps, samples)
