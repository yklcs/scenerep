import jax
import jax.numpy as jnp

from scenerep.models import RadianceField


class ExampleRadianceField(RadianceField):
    bounds = jnp.array([[-1, 1], [-1, 1], [-1, 1]])

    def __call__(self, x: jax.Array, d: jax.Array):
        batch_size = x.shape[0]
        color = jnp.tile(jnp.array([1, 0, 0]), (batch_size, 1))

        density = jnp.exp(jnp.sum(-(x**2), axis=1))

        return color, density
