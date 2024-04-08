from jax import Array
import jax.numpy as jnp
import flax.linen as nn

from scenerep.models import RadianceField

from typing import Annotated


def lerp(a, b, t):
    return (1 - t) * a + t * b


class Plane(nn.Module):
    width: int
    height: int
    depth: int

    @staticmethod
    def interpolate(plane, x):
        """Bilinear interpolation"""
        x_frac, x_whole = jnp.modf(x)
        tx, ty = x_frac.T
        x0, y0 = x_whole.astype(int).T  # corner
        x1, y1 = x0 + 1, y0 + 1  #  opposite corner

        top = lerp(plane[x0, y0], plane[x1, y0], tx[..., jnp.newaxis])
        bottom = lerp(plane[x0, y1], plane[x1, y1], tx[..., jnp.newaxis])
        return lerp(top, bottom, ty[..., jnp.newaxis])

    @nn.compact
    def __call__(self, x):  # type: ignore
        plane = self.param(
            "plane", nn.initializers.uniform(), (self.width, self.height, self.depth)
        )
        z = self.interpolate(plane, x)
        return z


class PlanesRadianceField(RadianceField):
    bounds = jnp.array([[-1, 1], [-1, 1], [-1, 1]])

    def setup(self):
        self.mlp_color = [nn.Dense(32), nn.Dense(32), nn.Dense(3)]
        self.mlp_density = [nn.Dense(32), nn.Dense(32), nn.Dense(1)]

        self.planes_hi = (Plane(128, 128, 4), Plane(128, 128, 4), Plane(128, 128, 4))
        self.planes_lo = (Plane(32, 32, 4), Plane(32, 32, 4), Plane(32, 32, 4))

    def __call__(self, x: Annotated[Array, ("B", 3)], d: Annotated[Array, ("B", 3)]):
        xy, yz, zx = x[..., (0, 1)], x[..., (1, 2)], x[..., (2, 0)]

        z_hi = jnp.concat(
            (self.planes_hi[0](xy), self.planes_hi[1](yz), self.planes_hi[2](zx)),
            axis=-1,
        )
        z_lo = jnp.concat(
            (self.planes_lo[0](xy), self.planes_lo[1](yz), self.planes_lo[2](zx)),
            axis=-1,
        )

        z = jnp.concat((z_lo, z_hi, d), axis=-1)
        z_color = z_density = z

        for layer in self.mlp_color[:-1]:
            z_color = layer(z_color)
            z_color = nn.relu(z_color)
        color = nn.sigmoid(self.mlp_color[-1](z_color))

        for layer in self.mlp_density[:-1]:
            z_density = layer(z_density)
            z_density = nn.relu(z_density)
        density = nn.sigmoid(self.mlp_density[-1](z_density))

        return color, density
