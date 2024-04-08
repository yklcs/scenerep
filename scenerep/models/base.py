from abc import ABC, abstractmethod
from typing import Annotated

import flax.linen as nn
from jax import Array


class RadianceField(ABC, nn.Module):
    @abstractmethod
    def __call__(  # type: ignore
        self, x: Annotated[Array, ("B", 3)], d: Annotated[Array, ("B", 3)]
    ) -> tuple[Annotated[Array, ("B", 3)], Annotated[Array, ("B",)]]:
        """
        Radiance field function.
        Given batched positions and view directions, returns color and density.
        """
        ...
